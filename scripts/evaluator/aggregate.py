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

def update_flag(cfg, blend_cfg):
    mtbench_flag = jbbq_flag = lctg_flag = toxicity_flag = jtruthfulqa_flag = jaster_flag = GLP_flag = ALT_flag = False

    if hasattr(cfg, 'run'):
        mtbench_flag = cfg.run.mtbench
        jbbq_flag = cfg.run.jbbq
        lctg_flag = cfg.run.lctg
        toxicity_flag = cfg.run.toxicity
        jtruthfulqa_flag = cfg.run.jtruthfulqa
        jaster_flag = cfg.run.jaster

    if blend_cfg:
        for old_run in blend_cfg.old_runs:
            if old_run.dataset is None:
                continue
            for dataset in old_run.dataset:
                if "mtbench" in dataset:
                    mtbench_flag = True
                elif "jbbq" in dataset:
                    jbbq_flag = True
                elif "lctg" in dataset:
                    lctg_flag = True
                elif "toxicity" in dataset:
                    toxicity_flag = True
                elif "jtruthfulqa" in dataset:
                    jtruthfulqa_flag = True
                elif "jaster" in dataset:
                    jaster_flag = True

    if mtbench_flag and jaster_flag:
        GLP_flag = True
    if jbbq_flag and lctg_flag and toxicity_flag and jtruthfulqa_flag:
        ALT_flag = True
    return GLP_flag, ALT_flag


def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    blend_cfg = instance.blend_config
    num_few_shots = cfg.num_few_shots

    #GLP_flag, ALT_flag = update_flag(cfg, blend_cfg)
    GLP_flag= True
    ALT_flag = True

    # Initialize empty variables
    if GLP_flag or ALT_flag:
        jaster_0shot = jaster_fewshots = jmmlu_robust_fewshots = jaster_control_0shot = None
        jaster_control_fewshots = lctg_overall = jbbq_fewshots = toxicity = mtbench = None
        jaster_0shot = read_wandb_table(table_name=f"jaster_0shot_leaderboard_table", run=run)
        jaster_fewshots = read_wandb_table(table_name=f"jaster_{num_few_shots}shot_leaderboard_table", run=run)

    if GLP_flag:
        mtbench = read_wandb_table(table_name=f"mtbench_leaderboard_table", run=run)
    
    if ALT_flag:
        lctg_overall = read_wandb_table(table_name=f"lctg_overall_leaderboard_table", run=run)

        jmmlu_robust_fewshots = read_wandb_table(table_name=f"jmmlu_robust_{num_few_shots}shot_leaderboard_table", run=run)
        jaster_control_0shot = read_wandb_table(table_name=f"jaster_control_0shot_leaderboard_table", run=run)
        jaster_control_fewshots = read_wandb_table(table_name=f"jaster_control_{num_few_shots}shot_leaderboard_table", run=run)
        jbbq_fewshots = read_wandb_table(table_name=f"jbbq_{num_few_shots}shot_leaderboard_table", run=run)
        toxicity = read_wandb_table(table_name=f"toxicity_leaderboard_table", run=run)
        jtruthfulqa = read_wandb_table(table_name=f"jtruthfulqa_leaderboard_table", run=run)

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

    def create_subcategory_table(category, cols_jaster, cols_mtbench, other=None):
        table_name = f"subcategory_table_{category}"
        data = {}

        if other is None:
            data["model_name"]=cfg.model.pretrained_model_name_or_path
            data["AVG"] = calculate_combined_means(cols_jaster, cols_mtbench)
            if cols_jaster:
                for col in cols_jaster:
                    data[f"{col}_0shot"] =  jaster_0shot[col][0]
                    data[f"{col}_{num_few_shots}shot"] = jaster_fewshots[col][0]
            if cols_mtbench:
                for col in cols_mtbench:
                    data[f"{col}_mtbench"] = mtbench[col][0] / 10
        
        elif other == "control":
            data = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "AVG": np.mean([np.mean([jaster_control_0shot["AVG"][0], jaster_control_fewshots["AVG"][0]]), lctg_overall["AVG_Total_ctg"][0]]),
                "jaster_control_0shot":jaster_control_0shot["AVG"][0],
                "jaster_control_2shot":jaster_control_fewshots["AVG"][0],
                "lctg_avg_score": lctg_overall["AVG_Total_ctg"][0],
            }

        elif other == "toxicity":
            data = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "AVG": toxicity[["公平性", "社会規範", "禁止行為", "違反カテゴリ"]].values.mean(),
                "公平性": toxicity["公平性"][0],
                "社会規範": toxicity["社会規範"][0],
                "禁止行為": toxicity["禁止行為"][0],
                "違反カテゴリ": toxicity["違反カテゴリ"][0],
            }

        elif other == "bias":
            data = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "AVG": 1 - jbbq_fewshots["avg_abs_bias_score"][0],
                "abs_bias_score_fewshot": jbbq_fewshots["avg_abs_bias_score"][0],
            }

        elif other == "robust":
            data = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "AVG": jmmlu_robust_fewshots["robust_score"][0],
                "jmmlu_robust_fewshots": jmmlu_robust_fewshots["robust_score"][0],
            }

        elif other == "truthful":
            data = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "AVG": jtruthfulqa["overall_score"][0],
                "jtruthfulqa_overall_score": jtruthfulqa["overall_score"][0],
            }

        # Convert data to DataFrame
        subcategory_table = pd.DataFrame([data])
        run.log({table_name: wandb.Table(dataframe=subcategory_table)})

    def calculate_average_from_dict(data_dict, prefix):
        relevant_items = {key: value for key, value in data_dict.items() if key.startswith(prefix)}
        relevant_values = [value for value in relevant_items.values() if isinstance(value, (int, float))]
        if relevant_values:
            return sum(relevant_values) / len(relevant_values)
        return float('nan')

    leaderboard_dict = {}
    leaderboard_dict["model_name"] = cfg.model.pretrained_model_name_or_path
    leaderboard_dict["model_size_category"] = cfg.model.get("size_category", np.nan)
    leaderboard_dict["model_size"] = cfg.model.get("size", np.nan)
    leaderboard_dict["model_release_date"] = pd.to_datetime(cfg.model.release_date, format='%m/%d/%Y')
    first_cols = ["model_name","model_size_category"]
    
    if GLP_flag:
        leaderboard_dict["GLP_表現"] = calculate_combined_means([],["roleplay","writing","humanities"])
        create_subcategory_table("expression", [], ["roleplay","writing","humanities"])
        leaderboard_dict["GLP_翻訳"] = calculate_combined_means(["alt-e-to-j","alt-j-to-e","wikicorpus-e-to-j","wikicorpus-j-to-e"], [])
        create_subcategory_table("translation", ["alt-e-to-j","alt-j-to-e","wikicorpus-e-to-j","wikicorpus-j-to-e"], [])
        leaderboard_dict["GLP_情報検索"] = calculate_combined_means(["jsquad"], [])
        create_subcategory_table("information_extraction", ["jsquad"], [])
        leaderboard_dict["GLP_推論"] = calculate_combined_means([], ["reasoning"])
        create_subcategory_table("reasoning", [], ["reasoning"])
        leaderboard_dict["GLP_数学的推論"] = calculate_combined_means(["mawps","mgsm"], ["math"])
        create_subcategory_table("mathematical_reasoning", ["mawps", "mgsm"], ["math"])
        leaderboard_dict["GLP_抽出"] = calculate_combined_means(["wiki_ner", "wiki_coreference", "chabsa"], ["extraction"])
        create_subcategory_table("entity_extraction", ["wiki_ner", "wiki_coreference", "chabsa"], ["extraction"])
        leaderboard_dict["GLP_知識・質問応答"] = calculate_combined_means(["jcommonsenseqa","jemhopqa", "jmmlu","niilc","aio"], ["stem"])
        create_subcategory_table("knowledge_QA", ["jcommonsenseqa","jemhopqa", "jmmlu","niilc","aio"], ["stem"])
        leaderboard_dict["GLP_英語"] = calculate_combined_means(["mmlu_en"], [])
        create_subcategory_table("english", ["mmlu_en"], [])
        leaderboard_dict["GLP_意味解析"] = calculate_combined_means(["jnli","janli","jsem","jsick", "jamp"], [])
        create_subcategory_table("semantic_analysis", ["jnli","janli","jsem","jsick", "jamp"], [])
        leaderboard_dict["GLP_構文解析"] = calculate_combined_means(["jcola-in-domain","jcola-out-of-domain","jblimp","wiki_reading","wiki_pas","wiki_dependency"], [])   
        create_subcategory_table("syntactic_analysis", ["jcola-in-domain","jcola-out-of-domain","jblimp","wiki_reading","wiki_pas","wiki_dependency"], []) 
        leaderboard_dict["汎用的言語性能(GLP)_AVG"] = calculate_average_from_dict(leaderboard_dict, "GLP")
        first_cols.append("汎用的言語性能(GLP)_AVG")

    if ALT_flag:
        leaderboard_dict["ALT_制御性"] = np.mean([np.mean([jaster_control_0shot["AVG"][0], jaster_control_fewshots["AVG"][0]]), lctg_overall["AVG_Total_ctg"][0]])
        create_subcategory_table("controllability", [], [], "control")
        leaderboard_dict["ALT_倫理・道徳"] = jaster_fewshots["commonsensemoralja"][0] # use only fewshots result
        create_subcategory_table("ethics", ["commonsensemoralja"], [])
        leaderboard_dict["ALT_毒性"] = toxicity[["公平性", "社会規範", "禁止行為", "違反カテゴリ"]].values.mean() if 'toxicity' in locals() else np.nan
        create_subcategory_table("toxicity", [], [], "toxicity")
        leaderboard_dict["ALT_バイアス"] = 1 - jbbq_fewshots["avg_abs_bias_score"][0]
        create_subcategory_table("bias", [], [], "bias")
        leaderboard_dict["ALT_堅牢性"] = jmmlu_robust_fewshots["robust_score"][0]
        create_subcategory_table("robustness", [], [], "robust")
        leaderboard_dict["ALT_真実性"] = jtruthfulqa["overall_score"][0]
        create_subcategory_table("truthfulness", [], [], "truthful")
        leaderboard_dict["アラインメント(ALT)_AVG"] = calculate_average_from_dict(leaderboard_dict, "ALT")
        first_cols.append("アラインメント(ALT)_AVG")

    if GLP_flag and ALT_flag:
        leaderboard_dict["TOTAL_AVG"] = np.mean([leaderboard_dict["汎用的言語性能(GLP)_AVG"], leaderboard_dict["アラインメント(ALT)_AVG"]])
        first_cols.append("TOTAL_AVG")

    # Average of each dataset
    if GLP_flag or ALT_flag:
        jaster_agg_cols = [c for c in jaster_0shot if not c.startswith("jmmlu_") and c not in ["run_name", "model_name"]]
        leaderboard_dict["AVG_jaster_0shot"] = jaster_0shot[jaster_agg_cols].mean(axis=1)[0]
        leaderboard_dict[f"AVG_jaster_{num_few_shots}shots"] = jaster_fewshots[jaster_agg_cols].mean(axis=1)[0]
    
    if GLP_flag:
        leaderboard_dict["AVG_mtbench"] = mtbench["AVG_mtbench"][0]
    
    if ALT_flag:
        leaderboard_dict["AVG_lctg"] = lctg_overall["AVG_Total_ctg"][0]

    leaderboard_table = pd.DataFrame([leaderboard_dict])
    cols = leaderboard_table.columns
    new_cols = first_cols + [c for c in cols if c not in first_cols]
    leaderboard_table = leaderboard_table[new_cols]
    # Radar table

    glp_radar_table = pd.DataFrame(
        data=radar_contents(
            leaderboard_dict=leaderboard_dict,
            categories=[
                "GLP_情報検索",
                "GLP_推論",
                "GLP_数学的推論",
                "GLP_抽出",
                "GLP_知識・質問応答",
                "GLP_英語",
                "GLP_意味解析",
                "GLP_構文解析",
            ],
        ),
        columns=["category", "score"],
    ) if GLP_flag else pd.DataFrame()

    alt_radar_table = pd.DataFrame(
        data=radar_contents(
            leaderboard_dict=leaderboard_dict,
            categories=[
                "ALT_制御性",
                "ALT_倫理・道徳",
                "ALT_毒性",
                "ALT_バイアス",
                "ALT_堅牢性",
                "ALT_真実性",
            ],
        ),
        columns=["category", "score"],
    ) if ALT_flag else pd.DataFrame()

    run.log({
        "leaderboard_table": wandb.Table(dataframe=leaderboard_table),
        "glp_radar_table": wandb.Table(dataframe=glp_radar_table) if GLP_flag else None,
        "alt_radar_table": wandb.Table(dataframe=alt_radar_table) if ALT_flag else None
    })
    run.finish()