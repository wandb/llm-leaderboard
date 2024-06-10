import pandas as pd
from config_singleton import WandbConfigSingleton
from . import symbol_to_ABCD, incorrect_to_ABCD


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
    instance = WandbConfigSingleton()

    df_list = []
    use_cols = [
        "run_name",
        "model_name",
        "dataset",
        "task",
        "num_few_shots",
        "subset",
        "index",
        "expected_output",
        "output",
    ]

    # normal
    normal_df = df.query("~task.str.endswith('Choice')")
    df_list.append(normal_df[use_cols].rename(columns={"output": "normal_output"}))

    # symbol
    suffix = "_SymbolChoice"
    symbol_df = df.query(f"~task.str.endswith('{suffix}')")
    new_col_name = f"output{suffix}"
    symbol_df[new_col_name] = symbol_df["output"].apply(symbol_to_ABCD)
    df_list.append(symbol_df[[new_col_name]])

    # incorrect
    suffix = "_IncorrectChoice"
    incorrect_df = df.query(f"~task.str.endswith('{suffix}')")
    new_col_name = f"output{suffix}"
    incorrect_df[new_col_name] = incorrect_df["output"].apply(symbol_to_ABCD)
    df_list.append(incorrect_df[[new_col_name]])

    # evaluation
    output_df = pd.concat(df_list, axis=1)
    output_df["score"] = output_df.apply(eval_robustness, axis=1)

    if subset == "":
        leaderboard_table = pd.pivot_table(
            data=output_df,
            values="score",
            index=["run_name", "model_name"],
            columns="dataset",
            aggfunc="mean",
        ).reset_index()
    else:
        leaderboard_table=[]
    return output_df, leaderboard_table
