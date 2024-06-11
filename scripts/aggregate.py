from pathlib import Path
import os
import wandb
import pandas as pd
import numpy as np
from utils import read_wandb_table


from config_singleton import WandbConfigSingleton


def aggregate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    few_shots=cfg.num_few_shots
    jaster_0shot=read_wandb_table(table_name=f"jaster_0shot_leaderboard_table", run=run)
    jaster_fewshots=read_wandb_table(table_name=f"jaster_{few_shots}shot_leaderboard_table", run=run)
    jmmlu_robust_fewshots=read_wandb_table(table_name=f"jmmlu_robust_{few_shots}shot_leaderboard_table", run=run)
    jaster_control_0shot=read_wandb_table(table_name=f"jaster_control_0shot_leaderboard_table", run=run)
    jaster_control_fewshots=read_wandb_table(table_name=f"jaster_control_{few_shots}shot_leaderboard_table", run=run)
    lctg_overall=read_wandb_table(table_name=f"lctg_overall_leaderboard_table", run=run)
    jbbq_0shot=read_wandb_table(table_name=f"jbbq_0shot_leaderboard_table", run=run)
    jbbq_fewshots=read_wandb_table(table_name=f"jbbq_{few_shots}shot_leaderboard_table", run=run)
    toxicity=read_wandb_table(table_name=f"toxicity_leaderboard_table", run=run)
    mtbench=read_wandb_table(table_name=f"mtbench_leaderboard_table", run=run)
    
    print("-------- aggregating results ----------")

    # leaderboard_table
    def calculate_combined_means(cols_jaster, cols_mtbench):
        means = []
        if cols_jaster:  # Check if cols_jaster is not empty
            for col in cols_jaster:
                mean_value = (jaster_0shot[col][0] + jaster_fewshots[col][0]) / 2
                means.append(mean_value)

        if cols_mtbench:
            for col in cols_mtbench:
                means.append(mtbench[col][0]/10)
        return np.mean(means)
    
    def calculate_average_from_dict(data_dict, prefix):
        # Extract items with the specified prefix
        relevant_items = {key: value for key, value in data_dict.items() if key.startswith(prefix)}
        # Calculate the mean of these items' values
        relevant_values = [value for value in relevant_items.values() if isinstance(value, (int, float))]
        if relevant_values:
            return sum(relevant_values) / len(relevant_values)
        return float('nan')  # Return NaN if no relevant values

    leaderboard_table = {}

    leaderboard_table["model_release_date"] = cfg.model.release_date
    leaderboard_table["model_size"] = cfg.model.size
    leaderboard_table["GLP_expression"] = calculate_combined_means([],["roleplay","writing","humanities"])
    leaderboard_table["GLP_translation"] = calculate_combined_means(["alt-e-to-j","alt-j-to-e","wikicorpus-e-to-j","wikicorpus-j-to-e"], [])
    #leaderboard_table["GLP_summarization"] =
    leaderboard_table["GLP_information_extraction"] = calculate_combined_means(["jsquad"], [])
    leaderboard_table["GLP_reasoning"] = calculate_combined_means([], ["reasoning"])
    leaderboard_table["GLP_mathematical_reasoning"] = calculate_combined_means(["mawps", "jmmlu_stem"], ["math"])
    leaderboard_table["GLP_entity_extraction"] = calculate_combined_means(["wiki_ner", "wiki_coreference", "chabsa"], ["extraction"])
    leaderboard_table["GLP_knowledge_QA"] = calculate_combined_means(["jcommonsenseqa","jemhopqa","jmmlu_humanities","jmmlu_social_sciences","jmmlu_other","niilc","aio"], ["stem"])
    leaderboard_table["GLP_English_MMLU"] = calculate_combined_means(["mmlu_en"], [])
    leaderboard_table["GLP_semantic_analysis"] = calculate_combined_means(["jnli","janli","jsem","jsick", "jamp"], [])
    leaderboard_table["GLP_syntactic_analysis"] = calculate_combined_means(["jcola-in-domain","jcola-out-of-domain","jblimp","wiki_reading","wiki_pas","wiki_dependency"], [])
    leaderboard_table["ALT_controllability"] = np.mean([np.mean([jaster_control_0shot["AVG"][0], jaster_control_fewshots["AVG"][0]]), lctg_overall["Total-AVG-ctg"][0]])
    leaderboard_table["ALT_ethics_moral"] = calculate_combined_means(["commonsensemoralja"],[])
    leaderboard_table["ALT_toxicity"] = toxicity[["公平性", "社会規範", "禁止行為", "違反カテゴリ"]].values.mean() if 'toxicity' in locals() else np.nan
    leaderboard_table["ALT_bias"] = 1-np.mean([jbbq_0shot["avg_abs_bias_score"][0], jbbq_fewshots["avg_abs_bias_score"][0]])
    #leaderboard_table["ALT_truthfulness"] = 
    leaderboard_table["ALT_robustness"] = jmmlu_robust_fewshots["jaster"][0]
    leaderboard_table["GLP_AVG"] = calculate_average_from_dict(leaderboard_table,"GLP") 
    leaderboard_table["ALT_AVG"] = calculate_average_from_dict(leaderboard_table,"ALT")
    leaderboard_table["TOTAL_AVG"] = np.mean([leaderboard_table["GLP_AVG"], leaderboard_table["ALT_AVG"]])
    leaderboard_table = pd.DataFrame([leaderboard_table])
    run.log({"leaderboard_table":  wandb.Table(dataframe=leaderboard_table)})
    run.finish()

