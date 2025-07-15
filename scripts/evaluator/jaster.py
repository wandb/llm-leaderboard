import json
import time
from pathlib import Path
import numpy as np

import pandas as pd
from toolz import pipe
from tqdm import tqdm
import wandb

from config_singleton import WandbConfigSingleton
from .evaluate_utils import (
    apply_chat_template,
    get_few_shot_messages,
    jaster_metrics_dict,
    controllability_dict,
    task_to_sub_category,
    LLMAsyncProcessor,
    extract_answer_with_pattern,
    AnswerPatternId,
    normalize,
    text_formatter,
    evaluate_robustness,
)

def evaluate_n_shot(few_shots: bool):
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    # download dataset
    dataset_name = "jaster"
    artifact = run.use_artifact(cfg[dataset_name].artifacts_path, type="dataset")
    artifact_dir = artifact.download()
    dataset_dir = Path(artifact_dir) / cfg[dataset_name].dataset_dir
    if not dataset_dir.exists():
        print(f"skip {dataset_name} because it is not found in {artifact_dir}")
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    tasks = [
        "aio",
        "alt-e-to-j",
        "alt-j-to-e",
        "commonsensemoralja",
        "jamp",
        "janli",
        "jblimp",
        "jcola-in-domain",
        "jcola-out-of-domain",
        "jcommonsenseqa",
        "jemhopqa",
        "jhumaneval",
        "jnli",
        "jmmlu",
        "jsem",
        "jsick",
        "jsquad",
        "kuci",
        "mawps",
        "mgsm",
        "niilc",
        "mmlu_prox_ja"
    ]

    if cfg.run.jmmlu_robustness and few_shots:
        tasks.extend(["jmmlu_IncorrectChoice", "jmmlu_SymbolChoice"])
    
    # jhumaneval is 0-shot only
    if few_shots and "jhumaneval" in tasks:
        tasks.remove("jhumaneval")
    
    if few_shots:
        num_few_shots = cfg.get("num_few_shots", None)
        if (num_few_shots is None) or (num_few_shots == 0):
            return
    else:
        num_few_shots = 0

    evaluation_results = []
    for task in tasks:
        # execute evaluation
        for subset in ("test", "dev"):
            eval_matainfo = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "dataset": dataset_name,
                "task": task,
                "num_few_shots": num_few_shots,
                "subset": subset,
            }

            # read task data
            task_data_path = dataset_dir / subset / f"{task}.json"
            if subset == "dev" and task == "mgsm":
                task_data_path = dataset_dir / "train" / f"mgsm.json" # mgsm is not in the dev set
            if not task_data_path.exists():
                print(f"skip {task} because it is not found in {artifact_dir}")
                continue
            with task_data_path.open(encoding="utf-8") as f:
                task_data = json.load(f)

            # number of evaluation samples
            if cfg.testmode:
                test_max_num_samples = 1
                val_max_num_samples = 1
            else:
                test_max_num_samples = 100
                val_max_num_samples = 10

            if subset == "test":
                num_samples = test_max_num_samples
            elif subset == "dev":
                num_samples = val_max_num_samples
            samples = task_data["samples"][:num_samples]

            for idx, sample in enumerate(samples):
                inputs = []
                # compose messages
                messages = []

                # add fewshots samples
                if few_shots:
                    few_shot_messages = get_few_shot_messages(
                        target_dataset_path=task_data_path,
                        num_few_shots=num_few_shots,
                    )
                    messages.extend(few_shot_messages)

                # user input
                messages.append({"role": "user", "content": sample["input"]})

                # instruction message
                message_intro = "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。リクエストを適切に完了するための回答を記述してください。"
                
                instruction = "\n".join(
                    [message_intro, task_data["instruction"]]
                )

                # Add instruction message at the beginning
                first_content = messages[0]["content"]
                messages[0]["content"] = f"{instruction}\n\n{first_content}"

                # generate output
                prompt = apply_chat_template(messages=messages)
                y_pred = None
                y_true: str = pipe(sample["output"], normalize)
                
                # Handle metrics for all tasks
                metrics_list = (
                    ["code_exec_sandbox", "pylint_check"] if task == "jhumaneval" else
                    ["comet_wmt22"] if task in ["alt-j-to-e", "alt-e-to-j"] else
                    ["exact_match_figure"] if task in ["mawps", "mgsm"] else
                    task_data["metrics"]
                )
                
                # Add inputs only once per sample (for LLM processing)
                generator_config = {"max_tokens": cfg.jaster.override_max_tokens or task_data["output_length"]}
                inputs.extend([messages, generator_config])
                
                for metrics in metrics_list:
                    metrics_func: callable = jaster_metrics_dict[metrics]
                    control_task = task.replace("_IncorrectChoice", "").replace("_SymbolChoice", "")
                    control_method: str = controllability_dict[control_task].__name__
                    control_func: callable = controllability_dict[control_task]

                    # collect data
                    evaluation_results.append(
                        {
                            **eval_matainfo,
                            "index": idx,
                            "input": sample["input"],
                            "raw_output": None,  # to be filled
                            "output": None,  # to be filled
                            "expected_output": y_true,
                            "prompt": prompt,
                            "metrics": metrics,
                            "metrics_func": metrics_func,
                            "control_method": control_method,
                            "control_func": control_func,
                            "score": None,  # to be filled
                            "inputs": inputs,
                        }
                    )

    all_inputs = [er["inputs"] for er in evaluation_results]
    llm_ap = LLMAsyncProcessor(
        llm=llm,
        inputs=all_inputs,
    )
    results = llm_ap.get_results()

    # Process all results uniformly
    for response, evaluation_result in tqdm(zip(results, evaluation_results)):
        raw_output = response.content
        if evaluation_result["task"] == "jhumaneval":
            print(f"DEBUG: raw_output: {repr(raw_output)}")
        
        # For jhumaneval, don't split by \n\n to preserve code blocks
        if evaluation_result["task"] == "jhumaneval":
            y_pred: str = pipe(
                raw_output,
                lambda x: text_formatter(x, evaluation_result["task"]),
                lambda x: x.strip(),
                lambda x: x.strip("'").strip('"'),
                lambda x: x.strip(),
                normalize,
            )
        else:
            y_pred: str = pipe(
                raw_output,
                lambda x: text_formatter(x, evaluation_result["task"]),
                lambda x: x.split("\n\n")[0],
                lambda x: x.strip(),
                lambda x: x.strip("'").strip('"'),
                lambda x: x.strip(),
                normalize,
            )
        
        # Handle all tasks uniformly
            metrics_func = evaluation_result["metrics_func"]
        
        if evaluation_result["metrics"] in ["code_exec_sandbox", "pylint_check"]: #jhumaneval
            # These metrics expect lists of predictions and ground truths
            # Extract code from the response using CODE_OUTPUT_JP pattern
            print(f"DEBUG: Before extraction - y_pred: {repr(y_pred)}")
            print(f"DEBUG: raw_output for extraction: {repr(raw_output)}")
            
            # Extract code from raw_output directly for jhumaneval
            print(f"DEBUG: Using pattern: {AnswerPatternId.CODE_OUTPUT_JP}")
            extracted_code = extract_answer_with_pattern(
                                    raw_output, 
                                    AnswerPatternId.CODE_OUTPUT_JP,
                                    None
                                )
            print(f"DEBUG: After extraction from raw_output - extracted_code: {repr(extracted_code)}")
            
            # Manual regex test
            import re
            manual_match = re.search(r"```(?:\w+)?\s*\n?(.*?)\n?```", raw_output, re.DOTALL)
            if manual_match:
                manual_extracted = manual_match.group(1).strip()
                print(f"DEBUG: Manual regex extraction: {repr(manual_extracted)}")
            else:
                print(f"DEBUG: Manual regex extraction failed")
            
            y_pred = normalize(extracted_code).strip()
            print(f"DEBUG: Final y_pred: {repr(y_pred)}")
            
            y_preds = [y_pred]
            y_trues = [evaluation_result["expected_output"]]
            score = metrics_func(y_preds, y_trues)
        elif evaluation_result["metrics"] == "comet_wmt22":
            score = np.nan # will be evaluated later by loading model off from GPU
        else:
            score = metrics_func(y_pred, evaluation_result["expected_output"])
    
        control_func = evaluation_result["control_func"]
        control_score = control_func(y_pred)
        evaluation_result["raw_output"] = raw_output
        evaluation_result["output"] = y_pred
        evaluation_result["score"] = score
        evaluation_result["control_score"] = control_score
        del evaluation_result["metrics_func"], evaluation_result["control_func"], evaluation_result["inputs"]
        
    # Handle all tasks uniformly
    output_df = pd.DataFrame(evaluation_results)
    
    # Separate jhumaneval results for special logging
    jhumaneval_df = output_df[output_df["task"] == "jhumaneval"].copy()
    other_df = output_df[output_df["task"] != "jhumaneval"].copy()
        
    # Handle jhumaneval separately
    if not jhumaneval_df.empty:
        jhumaneval_df['sub_category'] = jhumaneval_df['task'].map(task_to_sub_category)
        
        # Separate dev and test tables for jhumaneval
        jhumaneval_dev_table = jhumaneval_df.query("subset == 'dev'")
        jhumaneval_test_table = jhumaneval_df.query("subset == 'test'")
        
        # Calculate average scores for jhumaneval leaderboard
        if not jhumaneval_test_table.empty:
            jhumaneval_leaderboard_table = pd.pivot_table(
                data=jhumaneval_test_table,
                values="score",
                index="model_name",
                columns="metrics",
                aggfunc="mean",
            ).reset_index()
        
        # Add average of the two metrics
        jhumaneval_leaderboard_table['AVG'] = jhumaneval_leaderboard_table[['code_exec_sandbox', 'pylint_check']].mean(axis=1)
        jhumaneval_leaderboard_table.drop(columns=["model_name"], inplace=True)
        jhumaneval_leaderboard_table.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)
        
        # Reorder columns for jhumaneval tables
        new_order=["model_name","task","index","input","raw_output","output","expected_output",
                   "prompt","score","control_score","metrics","control_method",
                   "dataset","num_few_shots","subset","sub_category"]
        jhumaneval_dev_table = jhumaneval_dev_table[new_order]
        jhumaneval_test_table = jhumaneval_test_table[new_order]
        
        # Log jhumaneval tables separately
        run.log(
            {
                "jhumaneval_output_table_dev": jhumaneval_dev_table,
                "jhumaneval_output_table": jhumaneval_test_table,
                "jhumaneval_leaderboard_table": jhumaneval_leaderboard_table,
            }
        )
    
    # Handle other tasks
    if not other_df.empty:
        output_df = other_df

        # log table
        if cfg.run.jmmlu_robustness and few_shots:
            output_robust_df = output_df[output_df["task"].str.contains("jmmlu")].copy()
            output_robust_df.loc[:,"sub_category"] = "robust"
        output_df = output_df[~output_df['task'].isin(['jmmlu_SymbolChoice', 'jmmlu_IncorrectChoice'])].copy()

        # group task to sub_category
        output_df = output_df.copy()  # Create a copy to avoid SettingWithCopyWarning
        output_df['sub_category'] = output_df['task'].map(task_to_sub_category)  
        dev_table = output_df.query("subset == 'dev'")
        test_table = output_df.query("subset == 'test'")
        
        leaderboard_table_control = pd.pivot_table(
            data=test_table,
            values="control_score",
            index="model_name",
            columns="task",
            aggfunc="mean",
        ).reset_index()

        #leaderboard_table['AVG'] = leaderboard_table.iloc[:, 2:].mean(axis=1) # calculate later in jaster_translation.py
        leaderboard_table_control.insert(0, 'AVG', leaderboard_table_control.iloc[:, 2:].mean(axis=1))
        leaderboard_table_control.drop(columns=["model_name"], inplace=True)
        leaderboard_table_control.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)
        
        new_order=["model_name","task","index","input","raw_output","output","expected_output",
                   "prompt","score","control_score","metrics","control_method",
                   "dataset","num_few_shots","subset","sub_category"]
        dev_table = dev_table[new_order]
        test_table = test_table[new_order]

        run.log(
            {
                f"{dataset_name}_{num_few_shots}shot_output_table_dev": dev_table,
                f"{dataset_name}_{num_few_shots}shot_output_table": test_table,
                #f"{dataset_name}_{num_few_shots}shot_leaderboard_table": leaderboard_table,  # log later in jaster_translation.py
                f"{dataset_name}_control_{num_few_shots}shot_leaderboard_table": leaderboard_table_control,
            }
        )
        

        if cfg.run.jmmlu_robustness and few_shots:
            # need to be updated
            dev_robust_table = output_robust_df.query("subset == 'dev'")
            test_robust_table= output_robust_df.query("subset == 'test'")
            dev_robust_table_for_log,_ = evaluate_robustness(subset="dev", df=dev_robust_table)
            test_robust_table_for_log, leaderboard_robust_table= evaluate_robustness(subset="test", df=test_robust_table)
            run.log(
            {
                f"jmmlu_robust_{num_few_shots}shot_output_table_dev": dev_robust_table_for_log,
                f"jmmlu_robust_{num_few_shots}shot_output_table": test_robust_table_for_log,
                f"jmmlu_robust_{num_few_shots}shot_leaderboard_table": leaderboard_robust_table
            }
        )
        
def evaluate():
    evaluate_n_shot(few_shots=False)
    evaluate_n_shot(few_shots=True)