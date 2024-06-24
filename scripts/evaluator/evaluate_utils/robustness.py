import pandas as pd
from config_singleton import WandbConfigSingleton
from . import symbol_to_ABCD, ABCD_to_symbol, incorrect_to_ABCD, ABCD_to_incorrect

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
    symbol_df[f"converted_output{symbol_suffix}"] = symbol_df[f"output{symbol_suffix}"].apply(symbol_to_ABCD)

    # incorrect
    incorrect_suffix = "_IncorrectChoice"
    incorrect_df = df[df["task"].str.endswith(incorrect_suffix)]
    incorrect_df = incorrect_df[use_cols + ["output"]].rename(columns={"output": f"output{incorrect_suffix}"})
    incorrect_df[f"converted_output{incorrect_suffix}"] = incorrect_df[f"output{incorrect_suffix}"].apply(incorrect_to_ABCD)

    # normal_dfにsymbolとincorrectの列を追加
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

            normal_df.loc[normal_row.name, f"input{symbol_suffix}"] = symbol_row["input"]
            normal_df.loc[normal_row.name, f"output{symbol_suffix}"] = symbol_row[f"output{symbol_suffix}"]
            normal_df.loc[normal_row.name, f"converted_output{symbol_suffix}"] = symbol_row[f"converted_output{symbol_suffix}"]
            normal_df.loc[normal_row.name, f"expected_output{symbol_suffix}"] = symbol_row["expected_output"]
            normal_df.loc[normal_row.name, f"input{incorrect_suffix}"] = incorrect_row["input"]
            normal_df.loc[normal_row.name, f"output{incorrect_suffix}"] = incorrect_row[f"output{incorrect_suffix}"]
            normal_df.loc[normal_row.name, f"converted_output{incorrect_suffix}"] = incorrect_row[f"converted_output{incorrect_suffix}"]
            normal_df.loc[normal_row.name, f"expected_output{incorrect_suffix}"] = incorrect_row["expected_output"]

    # スコアの計算
    normal_df["score"] = normal_df.apply(eval_robustness, axis=1)

    # 列のrename & 列の順番を並び替える
    normal_df = normal_df.rename(columns={"input": "input_normal","expected_output":"expected_output_normal"})
    new_order=["model_name","index","score",
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

        leaderboard_table = leaderboard_table.rename(columns={"jaster": "robust_score"})
    else:
        leaderboard_table = []
    

    return normal_df, leaderboard_table