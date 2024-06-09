from pathlib import Path
import os
import wandb
import pandas as pd
from utils import read_wandb_table
from config_singleton import WandbConfigSingleton

def aggregate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    few_shots=cfg.num_few_shots

    jaster_0shot=read_wandb_table(f"jaster_{few_shots}shot_leaderboard_table")
    jaster_fewshots= read_wandb_table(f"jaster_{few_shots}shot_leaderboard_table")
    jmmlu_robost_fewshots=read_wandb_table(f"jmmlu_robost_{few_shots}shot_leaderboard_table")
    jaster_control_fewshots=read_wandb_table(f"jaster_control_{few_shots}shot_leaderboard_table")
    lctg_overall_leaderboard_table=read_wandb_table(f"lctg_overall_leaderboard_table")
    jbbq_0shot_leaderboard_table=read_wandb_table(f"jbbq_0shot_leaderboard_table")
    jbbq_4shot_leaderboard_table=read_wandb_table(f"jbbq_{few_shots}shot_leaderboard_table")
    toxicity_leaderboard_table=read_wandb_table(f"toxicity_leaderboard_table")
    mtbench_leaderboard_table=read_wandb_table(f"mtbench_leaderboard_table")
    
    print("-------- aggregating results ----------")

    # leaderboard_table
    
    leaderboard_table = pd.DataFrame(columns=[
                                        "model_release_date",
                                        "model_size",
                                        "GLP_expression",
                                        "GLP_translation",
                                        "GLP_summarization",
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
                                        "ALT_truthfulness",
                                        "ALT_robustness",
                                        "GLP_AVG",
                                        "LT_AVG",
                                        "TOTAL_AVG"
                                    ])
    
    leaderboard_table["model_release_date"] = cfg.model.release_date
    leaderboard_table["model_size"] = cfg.model.size
    leaderboard_table["GLP_expression"] = 
    leaderboard_table["GLP_translation"] = 
    leaderboard_table["GLP_summarization"] =
    leaderboard_table["GLP_information_extraction"] =
    leaderboard_table["GLP_reasoning"] =
    leaderboard_table["GLP_mathematical_reasoning"] =
    leaderboard_table["GLP_entity_extraction"] =
    leaderboard_table["GLP_knowledge_QA"] =
    leaderboard_table["GLP_English_MMLU"] =
    leaderboard_table["GLP_semantic_analysis"] =
    leaderboard_table["GLP_syntactic_analysis"] =
    leaderboard_table["ALT_controllability"] =
    leaderboard_table["ALT_ethics_moral"] =
    leaderboard_table["ALT_toxicity"] = toxicity_leaderboard_table[["公平性", "社会規範", "禁止行為", "違反カテゴリ"]].values.mean()
    leaderboard_table["ALT_bias"] =
    leaderboard_table["ALT_truthfulness"] =
    leaderboard_table["ALT_robustness"] =
    leaderboard_table["GLP_AVG"] =
    leaderboard_table["LT_AVG"] =
    leaderboard_table["TOTAL_AVG"] = 

    
    run.log({"leaderboard_table": instance.table})
    run.finish()
