from pathlib import Path
import os
import wandb
from tqdm import tqdm
import pandas as pd
from config_singleton import WandbConfigSingleton
from .evaluate_utils import (
    apply_chat_template,
    get_generated_result_wo_header_footer,
    get_format_is_valid,
    get_keyword_is_valid,
    get_prohibited_word_is_valid,
    get_char_count_is_valid,
    quality_process_results,
    calculate_true_proportions,
    transform_data,
    LLMAsyncProcessor,
)

def evaluate():
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    #############################################
    # download dataset
    #############################################
    dataset_name = "lctg"
    artifact = run.use_artifact(cfg[dataset_name].artifacts_path, type="dataset")
    artifact_dir = artifact.download()
    dataset_dir = Path(artifact_dir) / cfg[dataset_name].dataset_dir
    if not dataset_dir.exists():
        print(f"skip {dataset_name} because it is not found in {artifact_dir}")
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    #############################################
    # initialization 
    #############################################
    tasks = ["summary","ad_text","pros_and_cons"]
    limitation_type_list = ["format", "char_count", "keyword", "prohibited_word"]
    total_summary = pd.DataFrame()
    columns = [
        'question_id', 'task', 'category', 'requirement', 'base_text', 'request',
        'output', 'preprocessed_output', 'ctg', 'qual'
    ]
    output_df = pd.DataFrame(columns=columns)

    for task in tasks:
        file = f"{artifact_dir}/lctg/{task}_prompts_v1.jsonl"
        master_df = pd.read_json(file, orient="records", lines=True)
        if cfg.testmode:
            master_df = master_df.head(2)
        else:
            master_df = master_df.head(30)
        #############################################
        # generation 
        #############################################
        all_results = {lmt_type: [] for lmt_type in limitation_type_list}
        for lmt_type in limitation_type_list:
            print(f"Perspective of controllability: {lmt_type}")
            sample_list = master_df[f"prompt_{lmt_type}"].tolist()
            inputs = []
            for sample in tqdm(sample_list):
                messages = []
                messages.append({"role": "user", "content": sample})
                prompt = apply_chat_template(messages=messages)
                generator_config = {"max_tokens": 3500}
                inputs.append([prompt, generator_config])

            llm_ap = LLMAsyncProcessor(
                llm=llm,
                inputs=inputs,
                # batch_size=256,  # APIの場合変える必要あり
            )
            results = llm_ap.get_results()
            all_results[lmt_type]=[message[0].content for message in results]
        data = {
            "generated_text_id": list(range(len(master_df["prompt_id"].tolist()))),
            "prompt_id":master_df["prompt_id"].tolist(),
            "base_text": master_df["base_text"].tolist(),
            "format_result": all_results["format"],
            "char_count_result": all_results["char_count"],
            "keyword_result": all_results["keyword"],
            "prohibited_word_result": all_results["prohibited_word"]
        }
        
        # log for debug
        result_df = pd.DataFrame(data)

        #############################################
        # process generated results 
        #############################################        
        ctg_col_list = ["format_result", "char_count_result", "keyword_result", "prohibited_word_result"]
        for ctg_col in ctg_col_list:
            result_df[f"{ctg_col}_wo_hf"] = get_generated_result_wo_header_footer(result_df[ctg_col].tolist(), task)

        #############################################
        # calculate score ctg  
        #############################################
        result_df2 = result_df[["prompt_id", "format_result", "format_result_wo_hf", "char_count_result", "char_count_result_wo_hf", "keyword_result", "keyword_result_wo_hf", "prohibited_word_result", "prohibited_word_result_wo_hf"]]
        df_ctg = pd.merge(master_df, result_df2, on="prompt_id")

        validation_functions = {
                'is_valid_format': get_format_is_valid,
                'is_valid_char_count': get_char_count_is_valid,
                "is_valid_keyword": get_keyword_is_valid,
                "is_valid_prohibited_word": get_prohibited_word_is_valid
            }
        
        for validation in validation_functions:
            df_ctg[validation] = df_ctg.apply(lambda x: validation_functions[validation](x), axis=1).astype(int)

        sums = {key: df_ctg[key].sum() for key in validation_functions.keys()}

        # Calculate the average score for each validation type
        ctg_scores = {key: sums[key] / len(df_ctg) for key in validation_functions.keys()}

        for key, score in ctg_scores.items():
            print(f"{key.replace('is_valid_', '').replace('_', ' ').title()}: {score:.3f}")

        common_columns = [
            "prompt_id", "base_text", "char_count", "char_count_result", "char_count_result_wo_hf",
            "is_valid_char_count", "keyword", "keyword_result", "keyword_result_wo_hf", "is_valid_keyword",
            "prohibited_word", "prohibited_word_result", "prohibited_word_result_wo_hf",
            "is_valid_prohibited_word", "format", "format_result", "format_result_wo_hf", "is_valid_format"
        ]
        columns = common_columns + (["title"] if "title" in df_ctg.columns else [])
        save_df = df_ctg[columns].copy()
        result_columns = ["format_result", "format_result_wo_hf",
                          "char_count_result", "char_count_result_wo_hf",
                          "keyword_result", "keyword_result_wo_hf",
                          "prohibited_word_result", "prohibited_word_result_wo_hf",
                        ]
        for col in result_columns:
            save_df[col] = save_df[col].apply(lambda x: x)

        #############################################
        # calculate scores quality
        #############################################
        df_quality = quality_process_results(result_df, task)
        quality_scores = calculate_true_proportions(df_quality)

    #############################################
    # Calculate scores
    #############################################

        # individual outputs
        merged_df = pd.merge(df_quality,save_df, on='prompt_id', how='left')
        output_df = transform_data(output_df, merged_df, task)

        # task base summary
        scores_list = list(ctg_scores.values())+quality_scores
        task_summary_table = pd.DataFrame(data=[scores_list], columns=["Format-ctg","C-count-ctg","Keyword-ctg","P-word-ctg","Format-qual","C-count-qual","Keyword-qual","P-word-qual"])
        task_summary_table["AVG-ctg"] = task_summary_table.apply(lambda row: (row["Format-ctg"] + row["C-count-ctg"] + row["Keyword-ctg"] + row["P-word-ctg"]) / 4, axis=1)
        task_summary_table["AVG-qual"] = task_summary_table.apply(lambda row: (row["Format-qual"] + row["C-count-qual"] + row["Keyword-qual"] + row["P-word-qual"]) / 4, axis=1)
        columns = ["AVG-ctg", "AVG-qual"] + [col for col in task_summary_table.columns if col not in ["AVG-ctg", "AVG-qual"]]
        task_summary_table = task_summary_table[columns]

        for col in list(task_summary_table.columns):
            task_summary_table[col] = task_summary_table[col].map(lambda f: "{:.3f}".format(f))
        wandb.log({f"lctg_{task}_leaderboard_table": task_summary_table})

        total_summary[f"{task}_AVG-ctg"]=pd.to_numeric(task_summary_table['AVG-ctg'], errors='coerce')
        total_summary[f"{task}_AVG-qual"]=pd.to_numeric(task_summary_table['AVG-qual'], errors='coerce')

    # total summary
    AVG_columns_ctg = [f"{task}_AVG-ctg" for task in tasks]
    total_summary["Total-AVG-ctg"] = total_summary[AVG_columns_ctg].mean(axis=1)
    AVG_columns_qual = [f"{task}_AVG-qual" for task in tasks]
    total_summary["Total-AVG-qual"] = total_summary[AVG_columns_qual].mean(axis=1)
    total_summary["AVG"] = (total_summary["Total-AVG-ctg"]+total_summary["Total-AVG-qual"])/2

    columns = ['AVG'] + [col for col in total_summary.columns if col != 'AVG']
    total_summary = total_summary[columns]

    wandb.log({"lctg_overall_leaderboard_table": total_summary})
    wandb.log({"lctg_output_table": output_df})
    
    """
    contents of wandb.log
    "lctg_overall_leaderboard_table": leaderboard_table,
        # model_name 
        # AVG : 全てのAVG
        # Total-AVG-ctg : summary, ad_text, pros_and_consのAVG-ctgの平均
        # Total-AVG-qual : summary, ad_text, pros_and_consのAVG-qualの平均
        # summary-AVG-ctg : summaryのCTGスコアの平均 (Format-ctg, C-count-ctg, Keyword-ctg, P-word-ctgの平均)
        # summary-AVG-qual : summaryの品質スコアの平均 (Format-qual, C-count-qual, Keyword-qual, P-word-qualの平均)
        # ad_text-AVG-ctg : ad_textのCTGスコアの平均 (Format-ctg, C-count-ctg, Keyword-ctg, P-word-ctgの平均)
        # ad_text-AVG-qual : ad_textの品質スコアの平均 (Format-qual, C-count-qual, Keyword-qual, P-word-qualの平均)
        # pros_and_cons-AVG-ctg : pros_and_consのCTGスコアの平均 (Format-ctg, C-count-ctg, Keyword-ctg, P-word-ctgの平均)
        # pros_and_cons-AVG-qual : pros_and_consの品質スコアの平均 (Format-qual, C-count-qual, Keyword-qual, P-word-qualの平均)

    "lctg_summary_leaderboard_table": summary_leaderboard_table,
        # AVG-ctg : CTGスコアの平均 (Format-ctg, C-count-ctg, Keyword-ctg, P-word-ctgの平均)
        # AVG-qual : 品質スコアの平均 (Format-qual, C-count-qual, Keyword-qual, P-word-qualの平均)
        # Format-ctg : FormatのCTGスコア
        # Format-qual : Formatの品質スコア
        # C-count-ctg : Char countのCTGスコア
        # C-count-qual : Char countの品質スコア
        # Keyword-ctg : KeywordのCTGスコア
        # Keyword-qual : Keywordの品質スコア
        # P-word-ctg : Prohibited wordのCTGスコア
        # P-word-qual : Prohibited wordの品質スコア
    "lctg_ad_text_leaderboard_table": ad_text_leaderboard_table,
        # AVG-ctg : CTGスコアの平均 (Format-ctg, C-count-ctg, Keyword-ctg, P-word-ctgの平均)
        # AVG-qual : 品質スコアの平均 (Format-qual, C-count-qual, Keyword-qual, P-word-qualの平均)
        # Format-ctg : FormatのCTGスコア
        # Format-qual : Formatの品質スコア
        # C-count-ctg : Char countのCTGスコア
        # C-count-qual : Char countの品質スコア
        # Keyword-ctg : KeywordのCTGスコア
        # Keyword-qual : Keywordの品質スコア
        # P-word-ctg : Prohibited wordのCTGスコア
        # P-word-qual : Prohibited wordの品質スコア
    "lctg_pros_and_cons_table": pros_and_cons_leaderboard_table,
        # AVG-ctg : CTGスコアの平均 (Format-ctg, C-count-ctg, Keyword-ctg, P-word-ctgの平均)
        # AVG-qual : 品質スコアの平均 (Format-qual, C-count-qual, Keyword-qual, P-word-qualの平均)
        # Format-ctg : FormatのCTGスコア
        # Format-qual : Formatの品質スコア
        # C-count-ctg : Char countのCTGスコア
        # C-count-qual : Char countの品質スコア
        # Keyword-ctg : KeywordのCTGスコア
        # Keyword-qual : Keywordの品質スコア
        # P-word-ctg : Prohibited wordのCTGスコア
        # P-word-qual : Prohibited wordの品質スコア
    "lctg_output_table": output_df,
        # modle_name
        # qustion_id
        # task
        # category
        # requirement
        # input
        # output
        # preprocessed
        # ctg 
        # qual

    """

            


