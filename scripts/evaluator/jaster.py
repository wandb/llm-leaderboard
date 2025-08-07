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
    jmmlu_dict,
    LLMAsyncProcessor,
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
        "chabsa",
        "commonsensemoralja",
        "jamp",
        "janli",
        "jblimp",
        "jcola-in-domain",
        "jcola-out-of-domain",
        "jcommonsenseqa",
        "jemhopqa",
        "jnli",
        "jsem",
        "jsick",
        "jsquad",
        #"jsts",
        "kuci",
        "mawps",
        #"mbpp",
        "mgsm",
        "niilc",
        "wiki_coreference",
        "wiki_dependency",
        "wiki_ner",
        "wiki_pas",
        "wiki_reading",
        "wikicorpus-e-to-j",
        "wikicorpus-j-to-e"
    ]
    tasks.extend(sorted({p.stem for p in dataset_dir.glob("**/mmlu_en_*.json")}))
    tasks.extend(sorted({p.stem for p in dataset_dir.glob("**/jmmlu*.json") if not p.stem.endswith("Choice")}))

    if few_shots:
        num_few_shots = cfg.get("num_few_shots", None)
        if (num_few_shots is None) or (num_few_shots == 0):
            return
    else:
        num_few_shots = 0

    if cfg.run.jmmlu_robustness and few_shots:
        tasks.extend(sorted({p.stem for p in dataset_dir.glob("**/jmmlu*.json") if p.stem.endswith("Choice")}))

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
            if not task_data_path.exists():
                print(f"skip {task} because it is not found in {artifact_dir}")
                continue
            with task_data_path.open(encoding="utf-8") as f:
                task_data = json.load(f)

            # number of evaluation samples
            if cfg.testmode:
                test_max_num_samples = 1
                val_max_num_samples = 1
            elif "wiki" in task:
                test_max_num_samples = 20
                val_max_num_samples = 5
            elif "mmlu" in task:
                test_max_num_samples = 5
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
                if "mmlu_en" in task:
                    message_intro = "The following text provides instructions for a certain task."
                else:
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
                #metrics: str = task_data["metrics"][0]
                metrics: str = (
                                "comet_wmt22" if task in ["alt-j-to-e", "wikicorpus-j-to-e"] else
                                "comet_wmt22" if task in ["alt-e-to-j", "wikicorpus-e-to-j"] else
                                "exact_match_figure" if task in ["mawps", "mgsm"] else
                                task_data["metrics"][0]
                                )
                metrics_func: callable = jaster_metrics_dict[metrics]
                control_task = "mmlu_en" if "mmlu_en" in task else "jmmlu" if "jmmlu" in task else task
                control_method: str = controllability_dict[control_task].__name__
                control_func: callable = controllability_dict[control_task]

                # max_tokens の優先順位: cfg.jaster.max_tokens > cfg.generator.max_tokens > デフォルト値
                base_max_tokens = cfg.jaster.get("max_tokens") or cfg.generator.get("max_tokens", 128)
                
                # reasoning設定を確認
                reasoning_config = cfg.generator.get("extra_body", {}).get("reasoning", {})
                reasoning_max_tokens = reasoning_config.get("max_tokens", None)
                reasoning_effort = reasoning_config.get("effort", None)
                
                # use_exact_output_length が true の場合、task_data["output_length"] を正確に使用（reasoningの有無に関わらず）
                if cfg.jaster.get("use_exact_output_length", False):
                    task_output_length = task_data.get("output_length", 128)
                    answer_tokens = task_output_length
                # use_output_length_as_min が true の場合、task_data["output_length"] を最小値として使用
                elif cfg.jaster.get("use_output_length_as_min", True):
                    task_output_length = task_data.get("output_length", 128)
                    # 回答用のトークン数を確保（最小でもtask_output_lengthの2倍程度）
                    answer_tokens = max(task_output_length * 2, 50, base_max_tokens)
                else:
                    answer_tokens = base_max_tokens
                
                # reasoning使用時は、全体のmax_tokensを適切に設定
                if reasoning_max_tokens or reasoning_effort:
                    # OpenRouterでは、reasoning.max_tokensは全体のmax_tokensの内数
                    # 全体のmax_tokens = reasoning用 + 回答用
                    if reasoning_max_tokens:
                        # reasoning.max_tokensが指定されている場合
                        max_tokens = reasoning_max_tokens + answer_tokens
                    elif reasoning_effort:
                        # effortが指定されている場合（後方互換性のため）
                        # デフォルトのreasoning用トークン数を設定
                        if reasoning_effort == "high":
                            default_reasoning_tokens = 8000
                        elif reasoning_effort == "medium":
                            default_reasoning_tokens = 4000
                        else:  # low
                            default_reasoning_tokens = 2000
                        max_tokens = default_reasoning_tokens + answer_tokens
                else:
                    # reasoningなしの場合
                    max_tokens = answer_tokens
                
                # デバッグ: 最初のサンプルでmax_tokens設定を表示
                if idx == 0:
                    print(f"\n--- Jaster Task: {task} ---")
                    print(f"Task output_length: {task_data.get('output_length', 'Not specified')}")
                    print(f"Base max_tokens: {base_max_tokens}")
                    if cfg.jaster.get("use_exact_output_length", False):
                        print(f"Mode: use_exact_output_length")
                    elif cfg.jaster.get("use_output_length_as_min", True):
                        print(f"Mode: use_output_length_as_min")
                    else:
                        print(f"Mode: use base_max_tokens")
                    print(f"Answer tokens: {answer_tokens}")
                    if reasoning_max_tokens:
                        print(f"Reasoning max_tokens: {reasoning_max_tokens}")
                    elif reasoning_effort:
                        print(f"Reasoning effort: {reasoning_effort}")
                    print(f"Final max_tokens: {max_tokens}")
                
                generator_config = {"max_tokens": max_tokens}
                inputs.extend([messages, generator_config])

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

    # デバッグ: 最初の5つの結果を確認
    for i, r in enumerate(results[:5]):
        print(f"\n--- Debug Jaster Output {i} ---")
        print(f"Content: {repr(r.content)}")
        print(f"Content length: {len(r.content)}")
        if not r.content:
            print("WARNING: Empty content detected!")

    for response, evaluation_result in tqdm(zip(results, evaluation_results)):
        raw_output = response.content
        y_pred: str = pipe(
            raw_output,
            lambda x: text_formatter(x, evaluation_result["task"]),
            lambda x: x.split("\n\n")[0],
            lambda x: x.strip(),
            lambda x: x.strip("'").strip('"'),
            lambda x: x.strip(),
            normalize,
        )
        metrics_func = evaluation_result["metrics_func"]
        control_func = evaluation_result["control_func"]
        if evaluation_result["metrics"] == "comet_wmt22":
            score = np.nan
        else:
            metrics_func = evaluation_result["metrics_func"]
            score = metrics_func(y_pred, evaluation_result["expected_output"])        
        control_score = control_func(y_pred)
        evaluation_result["raw_output"] = raw_output
        evaluation_result["output"] = y_pred
        evaluation_result["score"] = score
        evaluation_result["control_score"] = control_score
        del evaluation_result["metrics_func"], evaluation_result["control_func"], evaluation_result["inputs"]
        
    output_df = pd.DataFrame(evaluation_results)
    # group mmlu_en and jmmlu task category
    output_df["task"] = output_df["task"].apply(lambda x: "mmlu_en" if x.startswith("mmlu_en") else x)
    output_df['task'] = output_df['task'].apply(lambda x: jmmlu_dict.get(x, x))
    output_df['task'] = output_df['task'].apply(
                                    lambda task: 'jmmlu_SymbolChoice' if task.endswith('_SymbolChoice') 
                                    else 'jmmlu_IncorrectChoice' if task.endswith('_IncorrectChoice') 
                                    else task
                                    )

    # log table
    if cfg.run.jmmlu_robustness and few_shots:
        output_robust_df = output_df[output_df["task"].str.contains("jmmlu")].copy()
        output_robust_df.loc[:,"sub_category"] = "robust"
    output_df = output_df[~output_df['task'].isin(['jmmlu_SymbolChoice', 'jmmlu_IncorrectChoice'])]


    # group mmlu_en and jmmlu task
    output_df['sub_category'] = output_df['task'].map(task_to_sub_category)  
    dev_table = output_df.query("subset == 'dev'")
    test_table = output_df.query("subset == 'test'")

    # calculate later in jaster_translation.py
    #leaderboard_table = pd.pivot_table(
    #    data=test_table,
    #    values="score",
    #    index=["run_name", "model_name"],
    #    columns="task",
    #    aggfunc="mean",
    #).reset_index()

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