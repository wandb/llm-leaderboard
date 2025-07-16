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
    """
    評価フラグを更新し、どのベンチマークを実行するかを決定する
    
    Returns:
        GLP_flag: 汎用的言語性能評価を実行するか
        ALT_flag: アラインメント評価を実行するか
        swebench_flag: SWE-Bench評価を実行するか
        additional_flags: その他のベンチマークのフラグ辞書
    """
    mtbench_flag = jbbq_flag  = toxicity_flag = jtruthfulqa_flag = jaster_flag = GLP_flag = ALT_flag = False
    
    # 新しいベンチマークのフラグ
    arc_agi_2_flag = bfcl_flag = hle_flag = jhumaneval_flag = hallulens_flag = False
    jmmlu_pro_flag = jamc_qa_flag = m_ifeval_flag = False  # 将来追加予定

    if hasattr(cfg, 'run'):
        mtbench_flag = cfg.run.mtbench
        jbbq_flag = cfg.run.jbbq
        toxicity_flag = cfg.run.toxicity
        jtruthfulqa_flag = cfg.run.jtruthfulqa
        jaster_flag = cfg.run.jaster
        swebench_flag = cfg.run.get('swebench', False)
        arc_agi_2_flag = cfg.run.get('arc_agi_2', False)
        bfcl_flag = cfg.run.get('bfcl', False)
        hle_flag = cfg.run.get('hle', False)
        hallulens_flag = cfg.run.get('hallulens', False)

    if blend_cfg:
        for old_run in blend_cfg.old_runs:
            if old_run.dataset is None:
                continue
            for dataset in old_run.dataset:
                if "mtbench" in dataset:
                    mtbench_flag = True
                elif "jbbq" in dataset:
                    jbbq_flag = True
                elif "toxicity" in dataset:
                    toxicity_flag = True
                elif "jtruthfulqa" in dataset:
                    jtruthfulqa_flag = True
                elif "jaster" in dataset:
                    jaster_flag = True
                elif "swebench" in dataset:
                    swebench_flag = True
                elif "arc_agi_2" in dataset:
                    arc_agi_2_flag = True
                elif "bfcl" in dataset:
                    bfcl_flag = True
                elif "hle" in dataset:
                    hle_flag = True
                elif "hallulens" in dataset:
                    hallulens_flag = True

    if mtbench_flag and jaster_flag:
        GLP_flag = True
    if jbbq_flag and toxicity_flag and jtruthfulqa_flag:
        ALT_flag = True
    
    # 新しいフラグの追加
    additional_flags = {
        'arc_agi_2': arc_agi_2_flag,
        'bfcl': bfcl_flag,
        'hle': hle_flag,
        'jhumaneval': jaster_flag,  # jhuman_evalはjasterの一部
        'hallulens': hallulens_flag,
        'jmmlu_pro': jmmlu_pro_flag,  # 将来的にjasterの一部として追加
        'jamc_qa': jamc_qa_flag,
        'm_ifeval': m_ifeval_flag,
    }
    
    return GLP_flag, ALT_flag, swebench_flag, additional_flags


def load_benchmark_results(run, flag_name, table_name, additional_flags):
    """
    ベンチマーク結果を読み込む
    
    Args:
        run: W&B run object
        flag_name: フラグ名
        table_name: テーブル名
        additional_flags: フラグ辞書
    
    Returns:
        読み込んだ結果またはNone
    """
    if additional_flags.get(flag_name, False):
        try:
            return read_wandb_table(table_name=table_name, run=run)
        except:
            print(f"{flag_name} results not found, skipping {flag_name} aggregation")
            additional_flags[flag_name] = False
    return None


def calculate_combined_means(jaster_0shot, jaster_fewshots, mtbench, cols_jaster, cols_mtbench):
    """
    Jasterの0-shot/few-shotとMT-benchの結果を組み合わせて平均を計算
    """
    means = []
    if cols_jaster and jaster_0shot is not None and jaster_fewshots is not None:
        for col in cols_jaster:
            if col in jaster_0shot.columns and col in jaster_fewshots.columns:
                mean_value = (jaster_0shot[col][0] + jaster_fewshots[col][0]) / 2
                means.append(mean_value)

    if cols_mtbench and mtbench is not None:
        for col in cols_mtbench:
            if col in mtbench.columns:
                means.append(mtbench[col][0] / 10)
    
    return np.mean(means) if means else float('nan')


def calculate_hierarchical_average(leaderboard_dict, category_list):
    """階層的な平均値を計算する関数"""
    values = []
    for cat in category_list:
        if cat in leaderboard_dict and isinstance(leaderboard_dict[cat], (int, float)):
            if not np.isnan(leaderboard_dict[cat]):
                values.append(leaderboard_dict[cat])
    if values:
        return sum(values) / len(values)
    return float('nan')


def calculate_average_from_dict(data_dict, prefix):
    """指定されたプレフィックスで始まる項目の平均を計算"""
    relevant_items = {key: value for key, value in data_dict.items() if key.startswith(prefix)}
    relevant_values = [value for value in relevant_items.values() if isinstance(value, (int, float)) and not np.isnan(value)]
    if relevant_values:
        return sum(relevant_values) / len(relevant_values)
    return float('nan')


def create_subcategory_table(run, cfg, category, data_dict, table_name=None):
    """サブカテゴリーテーブルを作成してW&Bにログ"""
    if table_name is None:
        table_name = f"subcategory_table_{category}"
    
    # データがまだ辞書でない場合は、モデル名を追加
    if "model_name" not in data_dict:
        data_dict["model_name"] = cfg.model.pretrained_model_name_or_path
    
    subcategory_table = pd.DataFrame([data_dict])
    run.log({table_name: wandb.Table(dataframe=subcategory_table)})


def calculate_glp_scores(cfg, leaderboard_dict, jaster_0shot, jaster_fewshots, mtbench, 
                        additional_flags, arc_agi_2_result, hle_result, swebench_flag, 
                        swebench_result, jhumaneval_result, bfcl_result, run):
    """
    汎用的言語性能（GLP）のスコアを階層的に計算
    
    階層構造:
    1. 汎用的言語性能（GLP）
       ├── 応用的言語性能
       │   ├── 表現
       │   ├── 翻訳
       │   └── 情報検索
       ├── 推論能力
       │   ├── 抽象的推論（ARC-AGI-2）
       │   ├── 論理的推論
       │   └── 数学的推論
       ├── 知識・質問応答
       │   ├── 一般的知識
       │   └── 専門的知識（JMMLU, JMMLU-Pro(将来), HLE）
       ├── 基礎的言語性能
       │   ├── 意味解析
       │   └── 構文解析
       └── アプリケーション開発
           ├── コーディング（SWE-Bench, JHumanEval）
           └── 関数呼び出し（BFCL）
    """
    
    # ========== 第3層：個別カテゴリの計算 ==========
    
    # --- 応用的言語性能 ---
    # 表現
    leaderboard_dict["GLP_表現"] = calculate_combined_means(
        jaster_0shot, jaster_fewshots, mtbench, [], ["roleplay", "writing", "humanities"]
    )
    create_subcategory_table(run, cfg, "expression", {
        "AVG": leaderboard_dict["GLP_表現"],
        "roleplay_mtbench": mtbench["roleplay"][0] / 10 if mtbench is not None and "roleplay" in mtbench.columns else float('nan'),
        "writing_mtbench": mtbench["writing"][0] / 10 if mtbench is not None and "writing" in mtbench.columns else float('nan'),
        "humanities_mtbench": mtbench["humanities"][0] / 10 if mtbench is not None and "humanities" in mtbench.columns else float('nan'),
    })
    
    # 翻訳
    leaderboard_dict["GLP_翻訳"] = calculate_combined_means(
        jaster_0shot, jaster_fewshots, mtbench, ["alt-e-to-j", "alt-j-to-e"], []
    )
    create_subcategory_table(run, cfg, "translation", {
        "AVG": leaderboard_dict["GLP_翻訳"],
        "alt-e-to-j_0shot": jaster_0shot["alt-e-to-j"][0] if "alt-e-to-j" in jaster_0shot.columns else float('nan'),
        "alt-e-to-j_fewshot": jaster_fewshots["alt-e-to-j"][0] if "alt-e-to-j" in jaster_fewshots.columns else float('nan'),
        "alt-j-to-e_0shot": jaster_0shot["alt-j-to-e"][0] if "alt-j-to-e" in jaster_0shot.columns else float('nan'),
        "alt-j-to-e_fewshot": jaster_fewshots["alt-j-to-e"][0] if "alt-j-to-e" in jaster_fewshots.columns else float('nan'),
    })
    
    # 情報検索
    leaderboard_dict["GLP_情報検索"] = calculate_combined_means(
        jaster_0shot, jaster_fewshots, mtbench, ["jsquad"], []
    )
    create_subcategory_table(run, cfg, "information_extraction", {
        "AVG": leaderboard_dict["GLP_情報検索"],
        "jsquad_0shot": jaster_0shot["jsquad"][0] if "jsquad" in jaster_0shot.columns else float('nan'),
        "jsquad_fewshot": jaster_fewshots["jsquad"][0] if "jsquad" in jaster_fewshots.columns else float('nan'),
    })
    
    # --- 推論能力 ---
    # 抽象的推論
    if additional_flags.get('arc_agi_2', False) and arc_agi_2_result is not None:
        leaderboard_dict["GLP_抽象的推論"] = arc_agi_2_result["AVG"][0]
        create_subcategory_table(run, cfg, "abstract_reasoning", {
            "AVG": leaderboard_dict["GLP_抽象的推論"],
            "arc_agi_2_score": arc_agi_2_result["AVG"][0],
        })
    else:
        leaderboard_dict["GLP_抽象的推論"] = float('nan')
    
    # 論理的推論
    leaderboard_dict["GLP_論理的推論"] = calculate_combined_means(
        jaster_0shot, jaster_fewshots, mtbench, [], ["reasoning"]
    )
    create_subcategory_table(run, cfg, "logical_reasoning", {
        "AVG": leaderboard_dict["GLP_論理的推論"],
        "reasoning_mtbench": mtbench["reasoning"][0] / 10 if mtbench is not None and "reasoning" in mtbench.columns else float('nan'),
    })
    
    # 数学的推論
    leaderboard_dict["GLP_数学的推論"] = calculate_combined_means(
        jaster_0shot, jaster_fewshots, mtbench, ["mawps", "mgsm"], ["math"]
    )
    create_subcategory_table(run, cfg, "mathematical_reasoning", {
        "AVG": leaderboard_dict["GLP_数学的推論"],
        "mawps_0shot": jaster_0shot["mawps"][0] if "mawps" in jaster_0shot.columns else float('nan'),
        "mawps_fewshot": jaster_fewshots["mawps"][0] if "mawps" in jaster_fewshots.columns else float('nan'),
        "mgsm_0shot": jaster_0shot["mgsm"][0] if "mgsm" in jaster_0shot.columns else float('nan'),
        "mgsm_fewshot": jaster_fewshots["mgsm"][0] if "mgsm" in jaster_fewshots.columns else float('nan'),
        "math_mtbench": mtbench["math"][0] / 10 if mtbench is not None and "math" in mtbench.columns else float('nan'),
    })
    
    # --- 知識・質問応答 ---
    # 一般的知識（JamC-QAは将来追加予定）
    general_knowledge_jaster = ["jcommonsenseqa", "jemhopqa", "niilc", "aio"]
    general_knowledge_mtbench = ["stem"]
    leaderboard_dict["GLP_一般的知識"] = calculate_combined_means(
        jaster_0shot, jaster_fewshots, mtbench, general_knowledge_jaster, general_knowledge_mtbench
    )
    
    # 一般的知識のサブカテゴリーテーブル
    general_knowledge_dict = {"AVG": leaderboard_dict["GLP_一般的知識"]}
    for col in general_knowledge_jaster:
        if jaster_0shot is not None and col in jaster_0shot.columns:
            general_knowledge_dict[f"{col}_0shot"] = jaster_0shot[col][0]
        if jaster_fewshots is not None and col in jaster_fewshots.columns:
            general_knowledge_dict[f"{col}_fewshot"] = jaster_fewshots[col][0]
    if mtbench is not None and "stem" in mtbench.columns:
        general_knowledge_dict["stem_mtbench"] = mtbench["stem"][0] / 10
    create_subcategory_table(run, cfg, "general_knowledge", general_knowledge_dict)
    
    # 専門的知識（JMMLU-Proは将来的にjasterの一部として追加予定）
    expert_knowledge_scores = []
    expert_knowledge_dict = {}
    
    if "jmmlu" in jaster_0shot.columns:
        jmmlu_score = (jaster_0shot["jmmlu"][0] + jaster_fewshots["jmmlu"][0]) / 2
        expert_knowledge_scores.append(jmmlu_score)
        expert_knowledge_dict["jmmlu_0shot"] = jaster_0shot["jmmlu"][0]
        expert_knowledge_dict["jmmlu_fewshot"] = jaster_fewshots["jmmlu"][0]
        expert_knowledge_dict["jmmlu_avg"] = jmmlu_score
    
    if hle_result is not None and additional_flags.get('hle', False):
        hle_score_raw = hle_result["accuracy"][0] if "accuracy" in hle_result.columns else float('nan')
        # HLEの精度はパーセント表記なので100で割る
        hle_score = hle_score_raw / 100.0 if not np.isnan(hle_score_raw) else float('nan')
        if not np.isnan(hle_score):
            expert_knowledge_scores.append(hle_score)
            expert_knowledge_dict["hle_accuracy"] = hle_score
        create_subcategory_table(run, cfg, "hle", {
            "AVG": hle_score,
            "accuracy": hle_score,
            "accuracy_raw": hle_score_raw,  # 元のパーセント値も保存
            "calibration_error": hle_result.get("calibration_error", [np.nan])[0],
            "confidence_half_width": hle_result.get("confidence_half_width", [np.nan])[0],
            "total_questions": hle_result.get("total_questions", [np.nan])[0],
            "answered_questions": hle_result.get("answered_questions", [np.nan])[0],
        })
    else:
        hle_score = float('nan')
    
    leaderboard_dict["GLP_専門的知識"] = np.mean(expert_knowledge_scores) if expert_knowledge_scores else float('nan')
    expert_knowledge_dict["AVG"] = leaderboard_dict["GLP_専門的知識"]
    create_subcategory_table(run, cfg, "expert_knowledge", expert_knowledge_dict)
    
    # --- 基礎的言語性能 ---
    # 意味解析
    semantic_jaster = ["jnli", "janli", "jsem", "jsick", "jamp"]
    leaderboard_dict["GLP_意味解析"] = calculate_combined_means(
        jaster_0shot, jaster_fewshots, mtbench, semantic_jaster, []
    )
    semantic_dict = {"AVG": leaderboard_dict["GLP_意味解析"]}
    for col in semantic_jaster:
        if jaster_0shot is not None and col in jaster_0shot.columns:
            semantic_dict[f"{col}_0shot"] = jaster_0shot[col][0]
        if jaster_fewshots is not None and col in jaster_fewshots.columns:
            semantic_dict[f"{col}_fewshot"] = jaster_fewshots[col][0]
    create_subcategory_table(run, cfg, "semantic_analysis", semantic_dict)
    
    # 構文解析
    syntactic_jaster = ["jcola-in-domain", "jcola-out-of-domain", "jblimp"]
    leaderboard_dict["GLP_構文解析"] = calculate_combined_means(
        jaster_0shot, jaster_fewshots, mtbench, syntactic_jaster, []
    )
    syntactic_dict = {"AVG": leaderboard_dict["GLP_構文解析"]}
    for col in syntactic_jaster:
        if jaster_0shot is not None and col in jaster_0shot.columns:
            syntactic_dict[f"{col}_0shot"] = jaster_0shot[col][0]
        if jaster_fewshots is not None and col in jaster_fewshots.columns:
            syntactic_dict[f"{col}_fewshot"] = jaster_fewshots[col][0]
    create_subcategory_table(run, cfg, "syntactic_analysis", syntactic_dict)
    
    # --- アプリケーション開発 ---
    # コーディング
    coding_scores = []
    coding_dict = {}
    
    if swebench_flag and swebench_result is not None:
        swebench_score = swebench_result["resolution_rate"][0]
        coding_scores.append(swebench_score)
        coding_dict["swebench_resolution_rate"] = swebench_score
        create_subcategory_table(run, cfg, "swebench", {
            "AVG": swebench_result["resolution_rate"][0],
            "resolution_rate": swebench_result["resolution_rate"][0],
            "issues_resolved": swebench_result["issues_resolved"][0],
            "total_samples": swebench_result["total_samples"][0],
            "application_rate": swebench_result.get("application_rate", ["N/A"])[0],
        })
    
    if additional_flags.get('jhumaneval', False) and jhumaneval_result is not None:
        jhumaneval_score = jhumaneval_result["AVG"][0]
        coding_scores.append(jhumaneval_score)
        coding_dict["jhumaneval_score"] = jhumaneval_score
    
    leaderboard_dict["GLP_コーディング"] = np.mean(coding_scores) if coding_scores else float('nan')
    coding_dict["AVG"] = leaderboard_dict["GLP_コーディング"]
    create_subcategory_table(run, cfg, "coding", coding_dict)
    
    # 関数呼び出し
    if additional_flags.get('bfcl', False) and bfcl_result is not None:
        leaderboard_dict["GLP_関数呼び出し"] = bfcl_result["Overall Acc"][0]
        create_subcategory_table(run, cfg, "function_calling", {
            "AVG": bfcl_result["Overall Acc"][0],
            "overall_accuracy": bfcl_result["Overall Acc"][0],
            "non_live_acc": bfcl_result.get("Non-Live Acc", [np.nan])[0],
            "live_acc": bfcl_result.get("Live Acc", [np.nan])[0],
            "multi_turn_acc": bfcl_result.get("Multi Turn Acc", [np.nan])[0],
            "irrelevance_detection": bfcl_result.get("Irrelevance Detection", [np.nan])[0],
        })
    else:
        leaderboard_dict["GLP_関数呼び出し"] = float('nan')
    
    # ========== 第2層：中間カテゴリの計算 ==========
    leaderboard_dict["GLP_応用的言語性能"] = calculate_hierarchical_average(
        leaderboard_dict, ["GLP_表現", "GLP_翻訳", "GLP_情報検索"]
    )
    leaderboard_dict["GLP_推論能力"] = calculate_hierarchical_average(
        leaderboard_dict, ["GLP_抽象的推論", "GLP_論理的推論", "GLP_数学的推論"]
    )
    leaderboard_dict["GLP_知識・質問応答"] = calculate_hierarchical_average(
        leaderboard_dict, ["GLP_一般的知識", "GLP_専門的知識"]
    )
    leaderboard_dict["GLP_基礎的言語性能"] = calculate_hierarchical_average(
        leaderboard_dict, ["GLP_意味解析", "GLP_構文解析"]
    )
    leaderboard_dict["GLP_アプリケーション開発"] = calculate_hierarchical_average(
        leaderboard_dict, ["GLP_コーディング", "GLP_関数呼び出し"]
    )
    
    # ========== 第1層：GLP全体の計算 ==========
    leaderboard_dict["汎用的言語性能(GLP)_AVG"] = calculate_hierarchical_average(
        leaderboard_dict, 
        ["GLP_応用的言語性能", "GLP_推論能力", "GLP_知識・質問応答", 
         "GLP_基礎的言語性能", "GLP_アプリケーション開発"]
    )


def calculate_alt_scores(cfg, leaderboard_dict, jaster_control_0shot, jaster_control_fewshots, 
                        jaster_fewshots, toxicity, jbbq_fewshots, jtruthfulqa, 
                        jmmlu_robust_fewshots, additional_flags, hallulens_result, run):
    """
    アラインメント（ALT）のスコアを計算
    
    構造:
    - アラインメント（ALT）
      ├── 制御性（M-IFEVAL は将来追加予定）
      ├── 倫理・道徳
      ├── 毒性
      ├── バイアス
      ├── 真実性（JTruthfulQA, HalluLens）
      └── 堅牢性
    """
    
    # 制御性（M-IFEVALは将来追加予定）
    if (jaster_control_0shot is not None and "AVG" in jaster_control_0shot.columns and 
        jaster_control_fewshots is not None and "AVG" in jaster_control_fewshots.columns):
        control_scores = [jaster_control_0shot["AVG"][0], jaster_control_fewshots["AVG"][0]]
        leaderboard_dict["ALT_制御性"] = np.mean(control_scores)
        create_subcategory_table(run, cfg, "controllability", {
            "AVG": leaderboard_dict["ALT_制御性"],
            "jaster_control_0shot": jaster_control_0shot["AVG"][0],
            "jaster_control_fewshot": jaster_control_fewshots["AVG"][0],
        })
    else:
        leaderboard_dict["ALT_制御性"] = float('nan')
        print(f"Warning: Jaster Control data not available")
    
    # 倫理・道徳
    if jaster_fewshots is not None and "commonsensemoralja" in jaster_fewshots.columns:
        leaderboard_dict["ALT_倫理・道徳"] = jaster_fewshots["commonsensemoralja"][0]
        create_subcategory_table(run, cfg, "ethics", {
            "AVG": leaderboard_dict["ALT_倫理・道徳"],
            "commonsensemoralja_fewshot": jaster_fewshots["commonsensemoralja"][0],
        })
    else:
        leaderboard_dict["ALT_倫理・道徳"] = float('nan')
        print(f"Warning: commonsensemoralja data not available in jaster_fewshots")
    
    # 毒性
    if toxicity is not None:
        toxicity_score = toxicity[["公平性", "社会規範", "禁止行為", "違反カテゴリ"]].values.mean()
        leaderboard_dict["ALT_毒性"] = toxicity_score
        create_subcategory_table(run, cfg, "toxicity", {
            "AVG": toxicity_score,
            "公平性": toxicity["公平性"][0],
            "社会規範": toxicity["社会規範"][0],
            "禁止行為": toxicity["禁止行為"][0],
            "違反カテゴリ": toxicity["違反カテゴリ"][0],
        })
    else:
        leaderboard_dict["ALT_毒性"] = float('nan')
    
    # バイアス
    if jbbq_fewshots is not None and "avg_abs_bias_score" in jbbq_fewshots.columns and jbbq_fewshots["avg_abs_bias_score"][0] is not None:
        leaderboard_dict["ALT_バイアス"] = 1 - jbbq_fewshots["avg_abs_bias_score"][0]
        create_subcategory_table(run, cfg, "bias", {
            "AVG": leaderboard_dict["ALT_バイアス"],
            "abs_bias_score_fewshot": jbbq_fewshots["avg_abs_bias_score"][0],
        })
    else:
        leaderboard_dict["ALT_バイアス"] = float('nan')
        print(f"Warning: JBBQ data not available or avg_abs_bias_score is missing")
    
    # 真実性（HalluLensを含む）
    truthfulness_scores = []
    if jtruthfulqa is not None:
        truthfulness_scores.append(jtruthfulqa["overall_score"][0])
    if additional_flags.get('hallulens', False) and hallulens_result is not None:
        if "refusal_rate" in hallulens_result.columns:
            truthfulness_scores.append(hallulens_result["refusal_rate"][0])
            create_subcategory_table(run, cfg, "hallulens", {
                "AVG": hallulens_result["refusal_rate"][0],
                "refusal_rate": hallulens_result["refusal_rate"][0],
            })
    
    leaderboard_dict["ALT_真実性"] = (
        np.mean(truthfulness_scores) if truthfulness_scores 
        else jtruthfulqa["overall_score"][0] if jtruthfulqa is not None 
        else float('nan')
    )
    create_subcategory_table(run, cfg, "truthfulness", {
        "AVG": leaderboard_dict["ALT_真実性"],
        "jtruthfulqa_overall_score": jtruthfulqa["overall_score"][0] if jtruthfulqa is not None else float('nan'),
    })
    
    # 堅牢性
    if jmmlu_robust_fewshots is not None and "robust_score" in jmmlu_robust_fewshots.columns:
        leaderboard_dict["ALT_堅牢性"] = jmmlu_robust_fewshots["robust_score"][0]
        create_subcategory_table(run, cfg, "robustness", {
            "AVG": leaderboard_dict["ALT_堅牢性"],
            "jmmlu_robust_fewshots": jmmlu_robust_fewshots["robust_score"][0],
        })
    else:
        leaderboard_dict["ALT_堅牢性"] = float('nan')
        print(f"Warning: JMMLU Robust data not available or robust_score is missing")
    
    # アラインメント全体の平均
    leaderboard_dict["アラインメント(ALT)_AVG"] = calculate_average_from_dict(leaderboard_dict, "ALT")


def evaluate():
    """
    評価結果を集計するメイン関数
    """
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    blend_cfg = instance.blend_config
    num_few_shots = cfg.num_few_shots

    # フラグの更新
    GLP_flag, ALT_flag, swebench_flag, additional_flags = update_flag(cfg, blend_cfg)

    # ========== データの読み込み ==========
    # GLPとALT共通のデータ
    jaster_0shot = jaster_fewshots = jmmlu_robust_fewshots = None
    jaster_control_0shot = jaster_control_fewshots = None
    jbbq_fewshots = toxicity = mtbench = jtruthfulqa = None
    
    if GLP_flag or ALT_flag:
        jaster_0shot = read_wandb_table(table_name=f"jaster_0shot_leaderboard_table", run=run)
        jaster_fewshots = read_wandb_table(table_name=f"jaster_{num_few_shots}shot_leaderboard_table", run=run)

    # SWE-Bench
    swebench_result = None
    if swebench_flag:
        try:
            swebench_result = read_wandb_table(table_name="swebench_leaderboard_table", run=run)
        except:
            print("SWE-Bench results not found, skipping SWE-Bench aggregation")
            swebench_flag = False

    # GLP用データ
    if GLP_flag:
        mtbench = read_wandb_table(table_name=f"mtbench_leaderboard_table", run=run)
    
    # ALT用データ
    if ALT_flag:
        jmmlu_robust_fewshots = read_wandb_table(table_name=f"jmmlu_robust_{num_few_shots}shot_leaderboard_table", run=run)
        jaster_control_0shot = read_wandb_table(table_name=f"jaster_control_0shot_leaderboard_table", run=run)
        jaster_control_fewshots = read_wandb_table(table_name=f"jaster_control_{num_few_shots}shot_leaderboard_table", run=run)
        jbbq_fewshots = read_wandb_table(table_name=f"jbbq_{num_few_shots}shot_leaderboard_table", run=run)
        toxicity = read_wandb_table(table_name=f"toxicity_leaderboard_table", run=run)
        jtruthfulqa = read_wandb_table(table_name=f"jtruthfulqa_leaderboard_table", run=run)

    # 新しいベンチマークのデータ
    arc_agi_2_result = load_benchmark_results(run, 'arc_agi_2', "arc_agi_2_leaderboard_table", additional_flags)
    bfcl_result = load_benchmark_results(run, 'bfcl', "bfcl_leaderboard_table", additional_flags)
    hle_result = load_benchmark_results(run, 'hle', "hle_leaderboard_table", additional_flags)
    jhumaneval_result = load_benchmark_results(run, 'jhumaneval', "jhumaneval_leaderboard_table", additional_flags)
    hallulens_result = load_benchmark_results(run, 'hallulens', "hallulens_leaderboard_table", additional_flags)

    print("-------- aggregating results ----------")

    # ========== リーダーボードの構築 ==========
    leaderboard_dict = {}
    leaderboard_dict["model_name"] = cfg.model.pretrained_model_name_or_path
    leaderboard_dict["model_size_category"] = cfg.model.get("size_category", np.nan)
    leaderboard_dict["model_size"] = cfg.model.get("size", np.nan)
    leaderboard_dict["model_release_date"] = pd.to_datetime(cfg.model.release_date, format='%m/%d/%Y')
    
    # GLP（汎用的言語性能）の計算
    if GLP_flag:
        calculate_glp_scores(
            cfg, leaderboard_dict, jaster_0shot, jaster_fewshots, mtbench,
            additional_flags, arc_agi_2_result, hle_result, swebench_flag,
            swebench_result, jhumaneval_result, bfcl_result, run
        )
    
    # ALT（アラインメント）の計算
    if ALT_flag:
        calculate_alt_scores(
            cfg, leaderboard_dict, jaster_control_0shot, jaster_control_fewshots,
            jaster_fewshots, toxicity, jbbq_fewshots, jtruthfulqa,
            jmmlu_robust_fewshots, additional_flags, hallulens_result, run
        )
    
    # 総合スコア
    if GLP_flag and ALT_flag:
        leaderboard_dict["TOTAL_AVG"] = np.mean([
            leaderboard_dict["汎用的言語性能(GLP)_AVG"], 
            leaderboard_dict["アラインメント(ALT)_AVG"]
        ])

    # ========== データセット別の平均値 ==========
    if GLP_flag or ALT_flag:
        jaster_agg_cols = [c for c in jaster_0shot if not c.startswith("jmmlu_") and c not in ["run_name", "model_name"]]
        leaderboard_dict["AVG_jaster_0shot"] = jaster_0shot[jaster_agg_cols].mean(axis=1)[0]
        leaderboard_dict[f"AVG_jaster_{num_few_shots}shots"] = jaster_fewshots[jaster_agg_cols].mean(axis=1)[0]
    
    if GLP_flag:
        leaderboard_dict["AVG_mtbench"] = mtbench["AVG_mtbench"][0]

    if swebench_flag:
        leaderboard_dict["AVG_swebench"] = swebench_result["resolution_rate"][0]

    # ========== テーブルの作成とログ ==========
    
    # 基本情報の列
    first_cols = ["model_name", "model_size_category"]
    
    # 総合スコアの列
    if GLP_flag and ALT_flag:
        first_cols.append("TOTAL_AVG")
    
    # GLPの階層構造テーブルの作成
    if GLP_flag:
        glp_hierarchy_data = {
            "category": ["汎用的言語性能(GLP)"],
            "score": [leaderboard_dict["汎用的言語性能(GLP)_AVG"]],
            "level": [1],
            "parent": [None]
        }
        
        # 第2層のカテゴリ
        for cat in ["応用的言語性能", "推論能力", "知識・質問応答", "基礎的言語性能", "アプリケーション開発"]:
            glp_hierarchy_data["category"].append(cat)
            glp_hierarchy_data["score"].append(leaderboard_dict[f"GLP_{cat}"])
            glp_hierarchy_data["level"].append(2)
            glp_hierarchy_data["parent"].append("汎用的言語性能(GLP)")
        
        # 第3層のカテゴリ
        subcategories = {
            "応用的言語性能": ["表現", "翻訳", "情報検索"],
            "推論能力": ["抽象的推論", "論理的推論", "数学的推論"],
            "知識・質問応答": ["一般的知識", "専門的知識"],
            "基礎的言語性能": ["意味解析", "構文解析"],
            "アプリケーション開発": ["コーディング", "関数呼び出し"]
        }
        
        for parent, subs in subcategories.items():
            for sub in subs:
                glp_hierarchy_data["category"].append(sub)
                glp_hierarchy_data["score"].append(leaderboard_dict[f"GLP_{sub}"])
                glp_hierarchy_data["level"].append(3)
                glp_hierarchy_data["parent"].append(parent)
        
        glp_hierarchy_table = pd.DataFrame(glp_hierarchy_data)
    else:
        glp_hierarchy_table = None

    # GLPの階層的な列を順序立てて追加
    glp_cols = []
    if GLP_flag:
        # 第1層（最上位）
        glp_cols.append("汎用的言語性能(GLP)_AVG")
        
        # 第2層（中間カテゴリ）
        glp_cols.extend([
            "GLP_応用的言語性能",
            "GLP_推論能力", 
            "GLP_知識・質問応答",
            "GLP_基礎的言語性能",
            "GLP_アプリケーション開発"
        ])
        
        # 第3層（個別カテゴリ）
        # 応用的言語性能の詳細
        glp_cols.extend(["GLP_表現", "GLP_翻訳", "GLP_情報検索"])
        # 推論能力の詳細
        glp_cols.extend(["GLP_抽象的推論", "GLP_論理的推論", "GLP_数学的推論"])
        # 知識・質問応答の詳細
        glp_cols.extend(["GLP_一般的知識", "GLP_専門的知識"])
        # 基礎的言語性能の詳細
        glp_cols.extend(["GLP_意味解析", "GLP_構文解析"])
        # アプリケーション開発の詳細
        glp_cols.extend(["GLP_コーディング", "GLP_関数呼び出し"])
    
    # ALTの列を追加
    alt_cols = []
    if ALT_flag:
        # ALT総合
        alt_cols.append("アラインメント(ALT)_AVG")
        # ALTカテゴリ
        alt_cols.extend([
            "ALT_制御性", "ALT_倫理・道徳", "ALT_毒性",
            "ALT_バイアス", "ALT_真実性", "ALT_堅牢性"
        ])
    
    # データセット別の平均値の列
    dataset_cols = []
    if GLP_flag or ALT_flag:
        dataset_cols.extend(["AVG_jaster_0shot", f"AVG_jaster_{num_few_shots}shots"])
    if GLP_flag:
        dataset_cols.append("AVG_mtbench")
    if swebench_flag:
        dataset_cols.append("AVG_swebench")
    
    # すべての列を結合
    ordered_cols = first_cols + glp_cols + alt_cols + dataset_cols
    
    # その他の列（モデル情報など）を追加
    leaderboard_table = pd.DataFrame([leaderboard_dict])
    remaining_cols = [c for c in leaderboard_table.columns if c not in ordered_cols]
    final_cols = ordered_cols + remaining_cols
    
    # 存在する列のみを選択
    final_cols = [c for c in final_cols if c in leaderboard_table.columns]
    leaderboard_table = leaderboard_table[final_cols]
    
    # レーダーチャート用のテーブル
    glp_radar_table = pd.DataFrame(
        data=radar_contents(
            leaderboard_dict=leaderboard_dict,
            categories=[
                "GLP_応用的言語性能",
                "GLP_推論能力",
                "GLP_知識・質問応答",
                "GLP_基礎的言語性能",
                "GLP_アプリケーション開発",
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

    # W&Bにログ
    run.log({
        "leaderboard_table": wandb.Table(dataframe=leaderboard_table),
        "glp_radar_table": wandb.Table(dataframe=glp_radar_table) if GLP_flag else None,
        "alt_radar_table": wandb.Table(dataframe=alt_radar_table) if ALT_flag else None,
        "glp_hierarchy_table": wandb.Table(dataframe=glp_hierarchy_table) if GLP_flag else None
    })
    run.finish()