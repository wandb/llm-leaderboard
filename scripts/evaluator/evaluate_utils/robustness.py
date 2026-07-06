import pandas as pd
import re
import string


CHOICE_SYMBOLS = ("$", "&", "#", "@", "%", "!", "?", "~", "^", "*", "+", "=")


def _option_label_markers(text: str) -> list[str]:
    option_start = text.find("選項：")
    if option_start >= 0:
        block = text[option_start + len("選項：") :]
    else:
        block = text
    return [match.group(1) for match in re.finditer(r"(?:^|,)([A-Z])\.", block)]


def infer_choice_labels(input_text: str, expected_outputs: list[str] | None = None) -> list[str]:
    """Infer contiguous option labels while ignoring initials inside choices."""
    markers = _option_label_markers(input_text or "")
    labels: list[str] = []
    search_from = 0
    for expected in string.ascii_uppercase:
        found_at = None
        for idx in range(search_from, len(markers)):
            if markers[idx] == expected:
                found_at = idx
                break
        if found_at is None:
            break
        labels.append(expected)
        search_from = found_at + 1

    for output in expected_outputs or []:
        for token in split_choice_tokens(output):
            if token in string.ascii_uppercase and token not in labels:
                output_index = string.ascii_uppercase.index(token) + 1
                labels = list(string.ascii_uppercase[: max(output_index, len(labels))])

    return labels or list("ABCD")


def split_choice_tokens(value: object) -> list[str]:
    text = str(value or "").strip().upper()
    if not text:
        return []
    return [
        token.strip()
        for token in re.split(r"[,，、\s]+", text)
        if token.strip()
    ]


def convert_symbol_choice(output: object, labels: list[str]) -> str:
    tokens = split_choice_tokens(output)
    if len(tokens) != 1:
        return str(output or "").strip().upper()
    token = tokens[0]
    if token in labels:
        return token
    symbol_mapping = {
        CHOICE_SYMBOLS[idx]: label
        for idx, label in enumerate(labels)
        if idx < len(CHOICE_SYMBOLS)
    }
    return symbol_mapping.get(token, token)


def convert_incorrect_choice(output: object, labels: list[str]) -> str:
    tokens = split_choice_tokens(output)
    if not tokens:
        return str(output or "").strip().upper()
    if len(tokens) == 1 and tokens[0] in labels:
        return tokens[0]

    token_set = set(tokens)
    label_set = set(labels)
    if token_set.issubset(label_set):
        missing = [label for label in labels if label not in token_set]
        if len(missing) == 1:
            return missing[0]

    return ",".join(tokens)

def eval_robustness(row):
    matches = sum([
        row["output_normal"] == row["converted_output_IncorrectChoice"],
        row["output_normal"] == row["converted_output_SymbolChoice"],
        row["converted_output_IncorrectChoice"] == row["converted_output_SymbolChoice"]
    ])
    
    if matches == 3:
        return 1.0
    elif matches == 1:
        return 0.5
    else:
        return 0.0

def evaluate_robustness(subset: str, df: pd.DataFrame):
    use_cols = [
        "model_name",
        "dataset",
        "task",
        "num_few_shots",
        "subset",
        "index",
        "input",
        "expected_output"
    ]

    # normal
    normal_df = df[~df["task"].str.endswith('Choice')]
    normal_df = normal_df[use_cols + ["output"]].rename(columns={"output": "output_normal"})

    # symbol
    symbol_suffix = "_SymbolChoice"
    symbol_df = df[df["task"].str.endswith(symbol_suffix)]
    symbol_df = symbol_df[use_cols + ["output"]].rename(columns={"output": f"output{symbol_suffix}"})

    # incorrect
    incorrect_suffix = "_IncorrectChoice"
    incorrect_df = df[df["task"].str.endswith(incorrect_suffix)]
    incorrect_df = incorrect_df[use_cols + ["output"]].rename(columns={"output": f"output{incorrect_suffix}"})

    # normal_dfにsymbolとincorrectの列を追加
    normal_df["choice_labels"] = None
    normal_df[f"input{symbol_suffix}"] = None
    normal_df[f"output{symbol_suffix}"] = None
    normal_df[f"converted_output{symbol_suffix}"] = None
    normal_df[f"input{incorrect_suffix}"] = None
    normal_df[f"output{incorrect_suffix}"] = None
    normal_df[f"converted_output{incorrect_suffix}"] = None
    normal_df[f"score"] = None

    # taskごとにnormal, symbol, incorrectでデータを取り出し、normal_dfに結果を追加
    for task in normal_df["task"].unique():
        normal_task_df = normal_df[normal_df["task"] == task]
        symbol_task_df = symbol_df[symbol_df["task"] == (task + symbol_suffix)]
        incorrect_task_df = incorrect_df[incorrect_df["task"] == (task + incorrect_suffix)]

        # 同じタスクのnormal, symbol, incorrectを一行ずつ取り出し、normal_dfにoutputを追加
        assert len(normal_task_df) == len(symbol_task_df) == len(incorrect_task_df), f"incorrect data size: {task}, {len(normal_task_df)}, {len(symbol_task_df)}, {len(incorrect_task_df)}"
        for i in range(len(normal_task_df)):
            normal_row = normal_task_df.iloc[i]
            symbol_row = symbol_task_df.iloc[i]
            incorrect_row = incorrect_task_df.iloc[i]
            labels = infer_choice_labels(
                normal_row["input"],
                [
                    normal_row["expected_output"],
                    symbol_row["expected_output"],
                    incorrect_row["expected_output"],
                ],
            )

            normal_df.loc[normal_row.name, "choice_labels"] = ",".join(labels)
            normal_df.loc[normal_row.name, f"input{symbol_suffix}"] = symbol_row["input"]
            normal_df.loc[normal_row.name, f"output{symbol_suffix}"] = symbol_row[f"output{symbol_suffix}"]
            normal_df.loc[normal_row.name, f"converted_output{symbol_suffix}"] = convert_symbol_choice(
                symbol_row[f"output{symbol_suffix}"],
                labels,
            )
            normal_df.loc[normal_row.name, f"expected_output{symbol_suffix}"] = symbol_row["expected_output"]
            normal_df.loc[normal_row.name, f"input{incorrect_suffix}"] = incorrect_row["input"]
            normal_df.loc[normal_row.name, f"output{incorrect_suffix}"] = incorrect_row[f"output{incorrect_suffix}"]
            normal_df.loc[normal_row.name, f"converted_output{incorrect_suffix}"] = convert_incorrect_choice(
                incorrect_row[f"output{incorrect_suffix}"],
                labels,
            )
            normal_df.loc[normal_row.name, f"expected_output{incorrect_suffix}"] = incorrect_row["expected_output"]

    # スコアの計算
    normal_df["score"] = normal_df.apply(eval_robustness, axis=1)

    # 列のrename & 列の順番を並び替える
    normal_df = normal_df.rename(columns={"input": "input_normal","expected_output":"expected_output_normal"})
    new_order=["model_name","index","score","choice_labels",
               "input_normal","output_normal","expected_output_normal",
               "input_SymbolChoice","output_SymbolChoice","converted_output_SymbolChoice","expected_output_SymbolChoice",
               "input_IncorrectChoice","output_IncorrectChoice","converted_output_IncorrectChoice","expected_output_IncorrectChoice","dataset","task","num_few_shots","subset"
               ]
    normal_df = normal_df[new_order]

    # データの確認
    if subset == "test":
        leaderboard_table = pd.pivot_table(
            data=normal_df,
            values="score",
            index=["model_name"],
            columns="dataset",
            aggfunc="mean",
        ).reset_index()

        rename_map = {
            col: "robust_score"
            for col in leaderboard_table.columns
            if col != "model_name"
        }
        leaderboard_table = leaderboard_table.rename(columns=rename_map)
    else:
        leaderboard_table = []
    

    return normal_df, leaderboard_table
