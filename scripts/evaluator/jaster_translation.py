import pandas as pd
from utils import read_wandb_table
from config_singleton import WandbConfigSingleton
from .evaluate_utils import commet_score


def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    num_few_shots=cfg.num_few_shots
    dataset_name = "jaster"
    for i in [0,num_few_shots]:
        jaster_output_table_dev=read_wandb_table(f"{dataset_name}_{i}shot_output_table_dev", run=run)
        jaster_output_table_test=read_wandb_table(f"{dataset_name}_{i}shot_output_table", run=run)
        
        updated_output_table_dev = pd.DataFrame(add_comet_evaluation_result(jaster_output_table_dev.to_dict(orient='records')))
        updated_output_table_test = pd.DataFrame(add_comet_evaluation_result(jaster_output_table_test.to_dict(orient='records')))

        leaderboard_table = pd.pivot_table(
            data=updated_output_table_test,
            values="score",
            index="model_name",
            columns="task",
            aggfunc="mean",
        ).reset_index()

        leaderboard_table.drop(columns=["model_name"], inplace=True)
        leaderboard_table.insert(0, 'AVG', leaderboard_table.iloc[:, 2:].mean(axis=1))
        leaderboard_table.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)
    
        run.log(
            {
                f"{dataset_name}_{i}shot_output_table_dev": updated_output_table_dev,
                f"{dataset_name}_{i}shot_output_table": updated_output_table_test,
                f"{dataset_name}_{i}shot_leaderboard_table": leaderboard_table,
            }
        )

def add_comet_evaluation_result(evaluation_results):
    # インデックスとスコアを計算するためのリストを初期化
    indices = []
    commet_src = []
    commet_mt = []
    commet_ref = []

    # evaluation_resultsの各要素（辞書）をループ処理
    for i, result in enumerate(evaluation_results):
        if "comet_wmt22" in result["metrics"]:
            # 'comet_wmt22' を持つインデックスをリストに追加
            indices.append(i)
            commet_src.append(result["input"])
            commet_mt.append(result["output"])
            commet_ref.append(result["expected_output"])

    if indices:  # インデックスリストが空でない場合
        # comet_scoreを計算
        commet_scores = commet_score(commet_src, commet_mt, commet_ref)

        # 結果を更新
        for index, new_score in zip(indices, commet_scores):
            evaluation_results[index]["score"] = new_score

    return evaluation_results