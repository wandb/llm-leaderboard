import pandas as pd
from config_singleton import WandbConfigSingleton
from . import symbol_to_ABCD, incorrect_to_ABCD, ABCD_to_incorrect

def eval_robustness(row):
    score = 0.0
    if not str(row["normal_output"]) in {"A", "B", "C", "D"}:
        return score
    else:
        if row["normal_output"] == row["output_IncorrectChoice"]:
            score += 0.5
        if row["normal_output"] == row["output_SymbolChoice"]:
            score += 0.5
        return score

def evaluate_robustness(num_few_shots: int, subset: str, df: pd.DataFrame):
    use_cols = [
        "run_name",
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
    normal_df = normal_df[use_cols + ["output"]].rename(columns={"output": "normal_output"})

    # symbol
    symbol_suffix = "_SymbolChoice"
    symbol_df = df[df["task"].str.endswith(symbol_suffix)]
    symbol_df = symbol_df[use_cols + ["output"]].rename(columns={"output": f"output{symbol_suffix}"})
    symbol_df[f"output{symbol_suffix}"] = symbol_df[f"output{symbol_suffix}"].apply(symbol_to_ABCD)

    # incorrect
    incorrect_suffix = "_IncorrectChoice"
    incorrect_df = df[df["task"].str.endswith(incorrect_suffix)]
    incorrect_df = incorrect_df[use_cols + ["output"]].rename(columns={"output": f"output{incorrect_suffix}"})
    incorrect_df[f"output{incorrect_suffix}"] = incorrect_df[f"output{incorrect_suffix}"].apply(incorrect_to_ABCD)

    # normal_dfにsymbolとincorrectの列を追加
    normal_df[f"input{symbol_suffix}"] = None
    normal_df[f"output{symbol_suffix}"] = None
    normal_df[f"input{incorrect_suffix}"] = None
    normal_df[f"output{incorrect_suffix}"] = None
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

            normal_df.loc[normal_row.name, f"input{symbol_suffix}"] = symbol_row["input"]
            normal_df.loc[normal_row.name, f"output{symbol_suffix}"] = symbol_row[f"output{symbol_suffix}"]
            normal_df.loc[normal_row.name, f"input{incorrect_suffix}"] = incorrect_row["input"]
            normal_df.loc[normal_row.name, f"output{incorrect_suffix}"] = incorrect_row[f"output{incorrect_suffix}"]

    # スコアの計算
    normal_df["score"] = normal_df.apply(eval_robustness, axis=1)

    # scoreの計算のためのoutput_IncorrectChoiceを複数選択から単一の回答に変換しているので、元に戻す
    normal_df[f"output{incorrect_suffix}"] = normal_df[f"output{incorrect_suffix}"].apply(ABCD_to_incorrect)

    # データの確認
    if subset == "":
        leaderboard_table = pd.pivot_table(
            data=normal_df,
            values="score",
            index=["run_name", "model_name"],
            columns="dataset",
            aggfunc="mean",
        ).reset_index()
    else:
        leaderboard_table = []

    return normal_df, leaderboard_table