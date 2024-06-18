from pathlib import Path
import os
import concurrent.futures
import wandb
from tqdm import tqdm
import pandas as pd
import numpy as np

from config_singleton import WandbConfigSingleton
from .evaluate_utils import (
    apply_chat_template,
    get_generated_result_wo_header_footer,
    get_format_is_valid,
    get_keyword_is_valid,
    get_prohibited_word_is_valid,
    get_char_count_is_valid,
    transform_data,
    LLMAsyncProcessor,
)


def parallel_inference(inputs, lmt_type, llm):
    llm_ap = LLMAsyncProcessor(
        llm=llm,
        inputs=inputs,
        # batch_size=256,  # APIの場合変える必要あり
    )
    results = llm_ap.get_results()
    results = [message[0].content for message in results]

    return lmt_type, results

def radar_contents(leaderboard_dict, categories: list[str]) -> list[list[str, float]]:
    ret = []
    for cat in categories:
        ret.append([cat[4:], leaderboard_dict[cat][0]])
    return ret

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
        'question_id', 'task', 'category', 'requirement', 'base_text', 'request', 'prompt',
        'output', 'preprocessed_output', 'ctg', 'qual'
    ]
    output_df = pd.DataFrame(columns=columns)
    Format_ctg=[]
    C_count_ctg=[]
    Keyword_ctg=[]
    P_word_ctg=[]

    prompt_table_data = []
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

        # Generate samples for API parallel processing
        for lmt_type in limitation_type_list:
            print(f"Perspective of controllability: {lmt_type}")
            sample_list = master_df[f"prompt_{lmt_type}"].tolist()
            inputs = []
            for i, sample in tqdm(enumerate(sample_list)):
                messages = []
                messages.append({"role": "user", "content": sample})
                prompt = apply_chat_template(messages=messages)
                generator_config = {"max_tokens": 1500}
                inputs.append([messages, generator_config])
                prompt_table_data.append({
                    "question_id": master_df["prompt_id"][i],
                    "category": lmt_type,
                    "_prompt": prompt})

            llm_ap = LLMAsyncProcessor(
                llm=llm,
                inputs=inputs
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
            print(f"Processing {ctg_col}")
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
            df_ctg[validation] = df_ctg.apply(lambda x: validation_functions[validation](x) if x is not None else 0, axis=1)
            #df_ctg[validation] = df_ctg.apply(lambda x: validation_functions[validation](x), axis=1).astype(int)

        sums = {key: df_ctg[key].sum() for key in validation_functions.keys()}

        # Calculate the average score for each validation type
        ctg_scores = {key: sums[key] / len(df_ctg) for key in validation_functions.keys()}

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
        #df_quality = quality_process_results(result_df, task)
        #quality_scores = calculate_true_proportions(df_quality)

    #############################################
    # Calculate scores
    #############################################
        # individual outputs
        #merged_df = pd.merge(df_quality,save_df, on='prompt_id', how='left')
        output_df = transform_data(output_df, save_df, task)

        # task base summary
        #scores_list = list(ctg_scores.values())+quality_scores
        scores_list = list(ctg_scores.values())
        #task_summary_table = pd.DataFrame(data=[scores_list], columns=["Format_ctg","C_count_ctg","Keyword_ctg","P_word_ctg","Format-qual","C-count-qual","Keyword-qual","P-word-qual"])
        task_summary_table = pd.DataFrame(data=[scores_list], columns=["Format_ctg","C_count_ctg","Keyword_ctg","P_word_ctg"])
        task_summary_table["AVG_ctg"] = task_summary_table.apply(lambda row: (row["Format_ctg"] + row["C_count_ctg"] + row["Keyword_ctg"] + row["P_word_ctg"]) / 4, axis=1)
        #task_summary_table["AVG_qual"] = task_summary_table.apply(lambda row: (row["Format-qual"] + row["C-count-qual"] + row["Keyword-qual"] + row["P-word-qual"]) / 4, axis=1)
        #columns = ["AVG_ctg", "AVG_qual"] + [col for col in task_summary_table.columns if col not in ["AVG_ctg", "AVG_qual"]]
        columns = ["AVG_ctg"] + [col for col in task_summary_table.columns if col not in ["AVG_ctg"]]
        task_summary_table = task_summary_table[columns]
        task_summary_table.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)

        numeric_columns = task_summary_table.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            task_summary_table[col] = task_summary_table[col].apply(lambda x: round(x, 3))
        wandb.log({f"lctg_{task}_leaderboard_table": task_summary_table})

        total_summary[f"AVG_{task}_ctg"] = pd.to_numeric(task_summary_table['AVG_ctg'], errors='coerce')
        
        #total_summary[f"{task}_AVG_qual"]=pd.to_numeric(task_summary_table['AVG_qual'], errors='coerce')

        Format_ctg.append(task_summary_table["Format_ctg"].iloc[0])
        C_count_ctg.append(task_summary_table["C_count_ctg"].iloc[0])
        Keyword_ctg.append(task_summary_table["Keyword_ctg"].iloc[0])
        P_word_ctg.append(task_summary_table["P_word_ctg"].iloc[0])

    # total summary
    AVG_columns_ctg = [f"AVG_{task}_ctg" for task in tasks]
    total_summary["AVG_Format_ctg"] = np.mean(Format_ctg)
    total_summary["AVG_C_count_ctg"] = np.mean(C_count_ctg)
    total_summary["AVG_Keyword_ctg"] = np.mean(Keyword_ctg)
    total_summary["AVG_P_word_ctg"] = np.mean(P_word_ctg)
    total_summary.insert(0, 'AVG_Total_ctg', total_summary[AVG_columns_ctg].mean(axis=1))
    total_summary.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)

    numeric_columns = total_summary.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        total_summary[col] = total_summary[col].apply(lambda x: round(x, 3))
    #AVG_columns_qual = [f"{task}_AVG_qual" for task in tasks]
    #total_summary["Total-AVG_qual"] = total_summary[AVG_columns_qual].mean(axis=1)
    #total_summary["AVG"] = (total_summary["Total-AVG_ctg"]+total_summary["Total-AVG_qual"])/2
    #columns = ['AVG'] + [col for col in total_summary.columns if col != 'AVG']
    #total_summary = total_summary[columns]


    lctg_task_radar_table = total_summary[['AVG_summary_ctg','AVG_ad_text_ctg','AVG_pros_and_cons_ctg']]
    lctg_subtask_radar_table = total_summary[['AVG_Format_ctg','AVG_C_count_ctg','AVG_Keyword_ctg','AVG_P_word_ctg']] 


    lctg_task_radar_table = pd.DataFrame(
        data=radar_contents(
            leaderboard_dict=lctg_task_radar_table.to_dict('dict'),
            categories=['AVG_summary_ctg','AVG_ad_text_ctg','AVG_pros_and_cons_ctg'],
        ),
        columns=["category", "score"],
    )
    lctg_task_radar_table['category'] = lctg_task_radar_table['category'].replace({
        'summary_AVG-ctg': 'summary',
        'ad_text_AVG-ctg': 'ad_text',
        'pros_and_cons_AVG-ctg': 'pros_and_cons'
    })
    lctg_subtask_radar_table = pd.DataFrame(
        data=radar_contents(
            leaderboard_dict=lctg_subtask_radar_table.to_dict('dict'),
            categories=['AVG_Format_ctg','AVG_C_count_ctg','AVG_Keyword_ctg','AVG_P_word_ctg'],
        ),
        columns=["category", "score"],
    )

    lctg_subtask_radar_table['category'] = lctg_subtask_radar_table['category'].replace({
        'AVG_Format_ctg': 'format',
        'AVG_C_count_ctg': 'c_count',
        'AVG_Keyword_ctg': 'keyword',
        'AVG_P_word_ctg': 'p_word',
    })

    # add prompt to output table
    prompt_table = pd.DataFrame(prompt_table_data)
    output_df = output_df.merge(prompt_table, how="left", on=["question_id", "category"])
    output_df["prompt"] = output_df["_prompt"]
    output_df.drop("_prompt", axis=1, inplace=True)
    output_df.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)

    wandb.log({"lctg_overall_leaderboard_table": wandb.Table(dataframe=total_summary)})
    wandb.log({"lctg_output_table": wandb.Table(dataframe=output_df)})
    wandb.log({"lctg_task_radar_table": wandb.Table(dataframe=lctg_task_radar_table )})
    wandb.log({"lctg_subtask_radar_table": wandb.Table(dataframe=lctg_subtask_radar_table)})