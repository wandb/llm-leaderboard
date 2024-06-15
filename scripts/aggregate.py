from pathlib import Path
import os
import wandb
import pandas as pd
import numpy as np
from utils import read_wandb_table
from config_singleton import WandbConfigSingleton

def radar_contents(leaderboard_dict, categories: list[str]) -> list[list[str, float]]:
    ret = []
    for cat in categories:
        ret.append([cat[4:], leaderboard_dict[cat]])
    return ret

def aggregate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    num_few_shots = cfg.num_few_shots
    
    # Initialize empty variables
    if cfg.run.GLP or cfg.run.ALT:
        jaster_0shot = jaster_fewshots = jmmlu_robust_fewshots = jaster_control_0shot = None
        jaster_control_fewshots = lctg_overall = jbbq_0shot = jbbq_fewshots = toxicity = mtbench = None
        jaster_0shot = read_wandb_table(table_name=f"jaster_0shot_leaderboard_table", run=run)
        jaster_fewshots = read_wandb_table(table_name=f"jaster_{num_few_shots}shot_leaderboard_table", run=run)

    if cfg.run.GLP:
        mtbench = read_wandb_table(table_name=f"mtbench_leaderboard_table", run=run)
    
    if cfg.run.ALT:
        lctg_overall = read_wandb_table(table_name=f"lctg_overall_leaderboard_table", run=run)
        jmmlu_robust_fewshots = read_wandb_table(table_name=f"jmmlu_robust_{num_few_shots}shot_leaderboard_table", run=run)
        jaster_control_0shot = read_wandb_table(table_name=f"jaster_control_0shot_leaderboard_table", run=run)
        jaster_control_fewshots = read_wandb_table(table_name=f"jaster_control_{num_few_shots}shot_leaderboard_table", run=run)
        jbbq_0shot = read_wandb_table(table_name=f"jbbq_0shot_leaderboard_table", run=run)
        jbbq_fewshots = read_wandb_table(table_name=f"jbbq_{num_few_shots}shot_leaderboard_table", run=run)
        toxicity = read_wandb_table(table_name=f"toxicity_leaderboard_table", run=run)

    print("-------- aggregating results ----------")

    def calculate_combined_means(cols_jaster, cols_mtbench):
        means = []
        if cols_jaster:
            for col in cols_jaster:
                mean_value = (jaster_0shot[col][0] + jaster_fewshots[col][0]) / 2
                means.append(mean_value)

        if cols_mtbench:
            for col in cols_mtbench:
                means.append(mtbench[col][0] / 10)
        return np.mean(means)

    def calculate_average_from_dict(data_dict, prefix):
        relevant_items = {key: value for key, value in data_dict.items() if key.startswith(prefix)}
        relevant_values = [value for value in relevant_items.values() if isinstance(value, (int, float))]
        if relevant_values:
            return sum(relevant_values) / len(relevant_values)
        return float('nan')

    leaderboard_dict = {}
    leaderboard_dict["model_release_date"] = pd.to_datetime(cfg.model.release_date, format='%m/%d/%Y')
    leaderboard_dict["model_size"] = cfg.model.size
    avg_cols = []
    
    if cfg.run.GLP: 
        leaderboard_dict["GLP_expression"] = calculate_combined_means([],["roleplay","writing","humanities"])
        leaderboard_dict["GLP_translation"] = calculate_combined_means(["alt-e-to-j","alt-j-to-e","wikicorpus-e-to-j","wikicorpus-j-to-e"], [])
        leaderboard_dict["GLP_information_extraction"] = calculate_combined_means(["jsquad"], [])
        leaderboard_dict["GLP_reasoning"] = calculate_combined_means([], ["reasoning"])
        leaderboard_dict["GLP_mathematical_reasoning"] = calculate_combined_means(["mawps"], ["math"])
        leaderboard_dict["GLP_entity_extraction"] = calculate_combined_means(["wiki_ner", "wiki_coreference", "chabsa"], ["extraction"])
        leaderboard_dict["GLP_knowledge_QA"] = calculate_combined_means(["jcommonsenseqa","jemhopqa", "jmmlu","niilc","aio"], ["stem"])
        leaderboard_dict["GLP_English_MMLU"] = calculate_combined_means(["mmlu_en"], [])
        leaderboard_dict["GLP_semantic_analysis"] = calculate_combined_means(["jnli","janli","jsem","jsick", "jamp"], [])
        leaderboard_dict["GLP_syntactic_analysis"] = calculate_combined_means(["jcola-in-domain","jcola-out-of-domain","jblimp","wiki_reading","wiki_pas","wiki_dependency"], [])    
        leaderboard_dict["GLP_AVG"] = calculate_average_from_dict(leaderboard_dict, "GLP")
        avg_cols.append("GLP_AVG")
    
    if cfg.run.ALT:
        leaderboard_dict["ALT_controllability"] = np.mean([np.mean([jaster_control_0shot["AVG"][0], jaster_control_fewshots["AVG"][0]]), lctg_overall["AVG_Total_ctg"][0]])
        leaderboard_dict["ALT_ethics_moral"] = calculate_combined_means(["commonsensemoralja"],[])
        leaderboard_dict["ALT_toxicity"] = toxicity[["公平性", "社会規範", "禁止行為", "違反カテゴリ"]].values.mean() if 'toxicity' in locals() else np.nan
        leaderboard_dict["ALT_bias"] = 1-np.mean([jbbq_0shot["avg_abs_bias_score"][0], jbbq_fewshots["avg_abs_bias_score"][0]])
        leaderboard_dict["ALT_robustness"] = jmmlu_robust_fewshots["jaster"][0]
        leaderboard_dict["ALT_AVG"] = calculate_average_from_dict(leaderboard_dict, "ALT")
        avg_cols.append("ALT_AVG")

    if cfg.run.GLP and cfg.run.ALT:
        leaderboard_dict["TOTAL_AVG"] = np.mean([leaderboard_dict["GLP_AVG"], leaderboard_dict["ALT_AVG"]])
        avg_cols.append("TOTAL_AVG")

    # Average of each dataset
    if cfg.run.GLP or cfg.run.ALT:
        jaster_agg_cols = [c for c in jaster_0shot if not c.startswith("jmmlu_") and c not in ["run_name", "model_name"]]
        leaderboard_dict["AVG_jaster_0shot"] = jaster_0shot[jaster_agg_cols].mean(axis=1)[0]
        leaderboard_dict[f"AVG_jaster_{num_few_shots}shots"] = jaster_fewshots[jaster_agg_cols].mean(axis=1)[0]
    
    if cfg.run.GLP:
        leaderboard_dict["AVG_mtbench"] = mtbench["AVG_mtbench"][0]
    
    if cfg.run.ALT:
        leaderboard_dict["AVG_lctg"] = lctg_overall["AVG_Total_ctg"][0]

    leaderboard_table = pd.DataFrame([leaderboard_dict])
    cols = leaderboard_table.columns
    new_cols = avg_cols + [c for c in cols if c not in avg_cols]
    leaderboard_table = leaderboard_table[new_cols]
    # Radar table

    glp_radar_table = pd.DataFrame(
        data=radar_contents(
            leaderboard_dict=leaderboard_dict,
            categories=[
                "GLP_information_extraction",
                "GLP_reasoning",
                "GLP_mathematical_reasoning",
                "GLP_entity_extraction",
                "GLP_knowledge_QA",
                "GLP_English_MMLU",
                "GLP_semantic_analysis",
                "GLP_syntactic_analysis",
            ],
        ),
        columns=["category", "score"],
    ) if cfg.run.GLP else pd.DataFrame()

    alt_radar_table = pd.DataFrame(
        data=radar_contents(
            leaderboard_dict=leaderboard_dict,
            categories=[
                "ALT_controllability",
                "ALT_ethics_moral",
                "ALT_toxicity",
                "ALT_bias",
                "ALT_robustness",
            ],
        ),
        columns=["category", "score"],
    ) if cfg.run.ALT else pd.DataFrame()

    run.log({
        "leaderboard_table": wandb.Table(dataframe=leaderboard_table),
        "glp_radar_table": wandb.Table(dataframe=glp_radar_table) if cfg.run.GLP else None,
        "alt_radar_table": wandb.Table(dataframe=alt_radar_table) if cfg.run.ALT else None
    })
    run.finish()
