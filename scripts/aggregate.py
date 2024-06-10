from pathlib import Path
import os
import wandb
import pandas as pd
from utils import read_wandb_table
from config_singleton import WandbConfigSingleton

def calculate_combined_means(cols_jaster, cols_mtbench):
    means = []
    # jaster0とjasterfewshotsの共通の列の平均を計算
    for col in cols_jaster:
        if col in jaster_0shot.columns and col in jaster_fewshots.columns:
            mean_value = (jaster_0shot[col].mean() + jaster_fewshots[col].mean()) / 2
            means.append(mean_value)
        else:
            means.append(np.nan)  # 共通の列がない場合はNaNを追加
    combined_means = []
    # mtbenchの列と先ほどの平均値のリストの平均を計算
    for i, col in enumerate(cols_mtbench):
        if col in mtbench.columns:
            mtbench_mean = mtbench[col].mean()
            combined_mean = (means[i] + mtbench_mean) / 2 if not np.isnan(means[i]) else mtbench_mean
            combined_means.append(combined_mean)
        else:
            combined_means.append(np.nan)  # 列がない場合はNaNを追加
    return np.mean(combined_means)

def aggregate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    few_shots=cfg.num_few_shots

    jaster_0shot=read_wandb_table(f"jaster_0shot_leaderboard_table")
    jaster_fewshots= read_wandb_table(f"jaster_{few_shots}shot_leaderboard_table")
    jmmlu_robost_fewshots=read_wandb_table(f"jmmlu_robost_{few_shots}shot_leaderboard_table")
    jaster_control_0shot=read_wandb_table(f"jaster_control_0shot_leaderboard_table")
    jaster_control_fewshots=read_wandb_table(f"jaster_control_{few_shots}shot_leaderboard_table")
    lctg_overall=read_wandb_table(f"lctg_overall_leaderboard_table")
    jbbq_0shot=read_wandb_table(f"jbbq_0shot_leaderboard_table")
    jbbq_4shot=read_wandb_table(f"jbbq_{few_shots}shot_leaderboard_table")
    toxicity=read_wandb_table(f"toxicity_leaderboard_table")
    mtbench=read_wandb_table(f"mtbench_leaderboard_table")
    
    print("-------- aggregating results ----------")

    # leaderboard_table
    
    leaderboard_table = pd.DataFrame(columns=[
                                        "model_release_date",
                                        "model_size",
                                        "GLP_expression",
                                        "GLP_translation",
                                        #"GLP_summarization",
                                        "GLP_information_extraction",
                                        "GLP_reasoning",
                                        "GLP_mathematical_reasoning",
                                        "GLP_entity_extraction",
                                        "GLP_knowledge_QA",
                                        "GLP_English_MMLU",
                                        "GLP_semantic_analysis",
                                        "GLP_syntactic_analysis",
                                        "ALT_controllability",
                                        "ALT_ethics_moral",
                                        "ALT_toxicity",
                                        "ALT_bias",
                                        #"ALT_truthfulness",
                                        "ALT_robustness",
                                        "GLP_AVG",
                                        "ALT_AVG",
                                        "TOTAL_AVG"
                                    ])
    

    leaderboard_table["model_release_date"] = cfg.model.release_date
    leaderboard_table["model_size"] = cfg.model.size
    leaderboard_table["GLP_expression"] = calculate_combined_means([],["roleplay","writing","humanities"])
    leaderboard_table["GLP_translation"] = calculate_combined_means(["alt-e-to-j","alt-j-to-e","wikicorpus-e-to-j","wikicorpus-j-to-e"], [])
    #leaderboard_table["GLP_summarization"] =
    leaderboard_table["GLP_information_extraction"] = calculate_combined_means(["jsquad"], [])
    leaderboard_table["GLP_reasoning"] = calculate_combined_means([], ["reasoning"])
    leaderboard_table["GLP_mathematical_reasoning"] = calculate_combined_means(["mawps", "jmmlu_stem"], ["math"])
    leaderboard_table["GLP_entity_extraction"] = calculate_combined_means(["wiki_ner", "wiki_coreference", "chabsa"], ["extraction"])
    leaderboard_table["GLP_knowledge_QA"] = calculate_combined_means(["jcommonsenseqa","jemhopqa","jmmlu_humanities","jmmlu_social_sciences","jmmlu_other","niilc","aio"], [])
    leaderboard_table["GLP_English_MMLU"] = calculate_combined_means(["mmlu_en"], [])
    leaderboard_table["GLP_semantic_analysis"] = calculate_combined_means(["jnli","janli","jsem","jsick", "jamp"], [])
    leaderboard_table["GLP_syntactic_analysis"] = calculate_combined_means(["jcola-in-domain","jcola-out-of-domain","jblimp","wiki_reading","wiki_pas","wiki_dependency"], [])
    leaderboard_table["ALT_controllability"] = np.mean(np.mean([jaster_control_0shot["AVG"], jaster_control_fewshots["AVG"]]), lctg_overall_leaderboard_table[["Total-AVG-ctg"]])
    leaderboard_table["ALT_ethics_moral"] = calculate_combined_means(["commonsensemoralja"],[])
    leaderboard_table["ALT_toxicity"] = toxicity_leaderboard_table[["公平性", "社会規範", "禁止行為", "違反カテゴリ"]].values.mean() if 'toxicity_leaderboard_table' in locals() else np.nan
    leaderboard_table["ALT_bias"] = 1-np.mean(jbbq_0shot["avg_abs_bias_score"], jbbq_fewshots["avg_abs_bias_score"])
    #leaderboard_table["ALT_truthfulness"] = 
    leaderboard_table["ALT_robustness"] = jmmlu_robost_fewshots["jmmlu_robustness"]
    leaderboard_table["GLP_AVG"] = leaderboard_table.filter(like='GLP').mean()
    leaderboard_table["ALT_AVG"] = leaderboard_table.filter(like='ALT').mean()
    leaderboard_table["TOTAL_AVG"] = np.mean(leaderboard_table["GLP_AVG"], leaderboard_table["ALT_AVG"])

    
    run.log({"leaderboard_table": instance.table})
    run.finish()