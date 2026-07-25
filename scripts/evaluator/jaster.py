import json
import re
import time
from pathlib import Path
import numpy as np

import pandas as pd
from toolz import pipe
from tqdm import tqdm
import wandb
import weave

from config_singleton import WandbConfigSingleton
from .evaluate_utils import (
    apply_chat_template,
    get_few_shot_messages,
    jaster_metrics_dict,
    controllability_dict,
    task_to_sub_category,
    extract_answer_with_pattern,
    AnswerPatternId,
    normalize,
    text_formatter,
    evaluate_robustness,
)
from .evaluate_utils.llm_response_checkpoint import (
    LLMResponseCheckpointStore,
    default_checkpoint_root,
    run_checkpointed_batch,
)


def _to_plain_dict(value) -> dict:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf
        if not isinstance(value, dict):
            return OmegaConf.to_container(value, resolve=True) or {}
    except Exception:
        pass
    return dict(value)


def _get_override_max_tokens(cfg, dataset_name: str):
    try:
        dataset_cfg = getattr(cfg, dataset_name, {})
    except Exception:
        return None
    return _to_plain_dict(dataset_cfg).get("override_max_tokens")


def evaluate_n_shot(few_shots: bool):
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm
    if (
        hasattr(llm, "async_client")
        and not hasattr(llm, "_get_async_client")
    ):
        # A paid run can reach JASTER after this module was updated while its
        # process still has the pre-loop-local adapter loaded. Give each
        # independently executed shot phase a fresh client so it cannot inherit
        # an AsyncOpenAI client owned by an already closed event loop.
        from llm_inference_adapter import get_llm_inference_engine

        llm = get_llm_inference_engine()
        print(
            "JASTER legacy live-run isolation: using a fresh inference client "
            "for this shot phase",
            flush=True,
        )

    # download dataset
    dataset_name = "jaster"
    artifact = run.use_artifact(cfg[dataset_name].artifacts_path, type="dataset")
    artifact_dir = artifact.download()
    dataset_dir = Path(artifact_dir) / cfg[dataset_name].dataset_dir
    if not dataset_dir.exists():
        print(f"skip {dataset_name} because it is not found in {artifact_dir}")
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    configured_tasks = cfg[dataset_name].get("tasks", None)
    tasks = (
        list(configured_tasks)
        if configured_tasks is not None
        else ["tmmluplus", "jhumaneval"]
    )
    require_configured_task_files = configured_tasks is not None

    if cfg.run.get("tmmluplus_robustness", False) and few_shots:
        tasks.extend(["tmmluplus_IncorrectChoice", "tmmluplus_SymbolChoice"])
    
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
    all_inputs = []
    request_keys = []
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
                if require_configured_task_files:
                    raise FileNotFoundError(
                        f"JASTER required task file not found: {task_data_path}"
                    )
                print(
                    "Skipping legacy optional JASTER task file because no "
                    f"explicit jaster.tasks list was configured: {task_data_path}",
                    flush=True,
                )
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
                message_intro = cfg.get("jaster", {}).get(
                    "message_intro",
                    "以下是一項任務的說明，附帶的輸入提供了更多上下文。請撰寫適當的回答以完成該任務。"
                )
                
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
                generator_config = _to_plain_dict(getattr(cfg, "generator", {}))
                override_max_tokens = _get_override_max_tokens(cfg, dataset_name)
                generator_config["max_tokens"] = override_max_tokens or task_data["output_length"]
                request_index = len(all_inputs)
                all_inputs.append([messages, generator_config])
                request_keys.append(
                    f"{num_few_shots}shot:{task}:{subset}:{idx}"
                )
                
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
                            "task_output_length": task_data["output_length"],
                            "requested_max_tokens": generator_config["max_tokens"],
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "finish_reason": None,
                            "content_was_none": None,
                            "reasoning_content_len": None,
                            "reasoning_content_preview": None,
                            "raw_output_is_empty": None,
                            "request_index": request_index,
                        }
                    )

    if not all_inputs:
        raise RuntimeError(
            f"JASTER {num_few_shots}-shot produced no evaluation requests"
        )

    configured_checkpoint_dir = cfg[dataset_name].get("checkpoint_dir", None)
    checkpoint_root = (
        Path(configured_checkpoint_dir)
        if configured_checkpoint_dir
        else default_checkpoint_root(run, dataset_name)
    )
    checkpoint_store = LLMResponseCheckpointStore(
        checkpoint_root / f"{num_few_shots}shot",
        model_name=cfg.model.pretrained_model_name_or_path,
    )
    results = run_checkpointed_batch(
        llm=llm,
        inputs=all_inputs,
        keys=request_keys,
        checkpoint_store=checkpoint_store,
        label=f"JASTER {num_few_shots}-shot",
    )
    if len(results) != len(all_inputs):
        raise RuntimeError(
            f"JASTER {num_few_shots}-shot response count mismatch: "
            f"{len(results)}/{len(all_inputs)}"
        )

    # Process all results uniformly
    for evaluation_result in tqdm(evaluation_results):
        response = results[evaluation_result["request_index"]]
        raw_output = response.content
        
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
            # 1) Try Japanese code fence
            extracted_code = extract_answer_with_pattern(
                raw_output,
                AnswerPatternId.CODE_OUTPUT_JP,
                None,
            )
            # 2) Fallback to English code fence
            if not extracted_code.strip():
                extracted_code = extract_answer_with_pattern(
                    raw_output,
                    AnswerPatternId.CODE_OUTPUT_EN,
                    None,
                )
            # 3) Fallback to generic code fence via custom regex (first fenced block)
            if not extracted_code.strip():
                extracted_code = extract_answer_with_pattern(
                    raw_output,
                    AnswerPatternId.CUSTOM,
                    r"```(?:\w+)?\s*\n?([\s\S]*?)\n?```",
                )
            # 4) Fallback to function definition onwards if present (no fence)
            if not extracted_code.strip():
                m = re.search(r"(?s)(def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(.*?\):[\s\S]*?)$", raw_output)
                if m:
                    extracted_code = m.group(1)
            # 5) Last resort: use raw output as-is
            if not extracted_code.strip():
                extracted_code = raw_output

            # Do NOT normalize code to avoid accidental character conversions
            y_pred = extracted_code.strip()

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
        evaluation_result["prompt_tokens"] = response.prompt_tokens
        evaluation_result["completion_tokens"] = response.completion_tokens
        evaluation_result["finish_reason"] = response.finish_reason
        evaluation_result["content_was_none"] = response.content_was_none
        evaluation_result["reasoning_content_len"] = len(response.reasoning_content or "")
        evaluation_result["reasoning_content_preview"] = (response.reasoning_content or "")[:200]
        evaluation_result["raw_output_is_empty"] = (raw_output == "")
        del (
            evaluation_result["metrics_func"],
            evaluation_result["control_func"],
            evaluation_result["request_index"],
        )
        
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
                   "task_output_length","requested_max_tokens","prompt_tokens","completion_tokens",
                   "finish_reason","content_was_none","reasoning_content_len","reasoning_content_preview",
                   "raw_output_is_empty",
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
        robust_task_prefix = None
        robust_table_prefix = None
        if cfg.run.get("tmmluplus_robustness", False) and few_shots:
            robust_task_prefix = "tmmluplus"
            robust_table_prefix = "tmmluplus_robust"
        elif cfg.run.get("jmmlu_robustness", False) and few_shots:
            robust_task_prefix = "jmmlu"
            robust_table_prefix = "jmmlu_robust"

        if robust_task_prefix:
            output_robust_df = output_df[
                output_df["task"].str.contains(robust_task_prefix)
            ].copy()
            output_robust_df.loc[:,"sub_category"] = "robust"
        excluded_variant_tasks = [
            "jmmlu_SymbolChoice",
            "jmmlu_IncorrectChoice",
            "tmmluplus_SymbolChoice",
            "tmmluplus_IncorrectChoice",
        ]
        output_df = output_df[~output_df['task'].isin(excluded_variant_tasks)].copy()

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
                   "task_output_length","requested_max_tokens","prompt_tokens","completion_tokens",
                   "finish_reason","content_was_none","reasoning_content_len","reasoning_content_preview",
                   "raw_output_is_empty",
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
        

        if robust_task_prefix:
            # need to be updated
            dev_robust_table = output_robust_df.query("subset == 'dev'")
            test_robust_table= output_robust_df.query("subset == 'test'")
            dev_robust_table_for_log,_ = evaluate_robustness(subset="dev", df=dev_robust_table)
            test_robust_table_for_log, leaderboard_robust_table= evaluate_robustness(subset="test", df=test_robust_table)
            run.log(
            {
                f"{robust_table_prefix}_{num_few_shots}shot_output_table_dev": dev_robust_table_for_log,
                f"{robust_table_prefix}_{num_few_shots}shot_output_table": test_robust_table_for_log,
                f"{robust_table_prefix}_{num_few_shots}shot_leaderboard_table": leaderboard_robust_table
            }
        )

@weave.op(call_display_name=lambda _: "[BFCL] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    evaluate_n_shot(few_shots=False)
    evaluate_n_shot(few_shots=True)
