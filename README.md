# Nejumi-leaderboard Neo

## Set up
0. Please use python 3.9 or 3.10 which are compatible for llm-jp-eval.
1. Set up environment variables
```
export WANDB_API_KEY=<your WANDB_API_KEY>
export OPENAI_API_KEY=<your OPENAI_API_KEY>
export LANG=ja_JP.UTF-8
```



## Data Prepartion 
The dataset has been already prepared in wandb artifacts, and the path is implemented into the default config template.
Evaluators can use them as they are.

- v 1.0.0: "wandb-japan/llm-leaderboard/jaster:v0"
- v 1.1.0: "wandb-japan/llm-leaderboard/jaster:v3"
- v 1.2.1: "wandb-japan/llm-leaderboard/jaster:v6"
- v 1.2.2: "wandb-japan/llm-leaderboard/jaster:v7"
- v 1.2.3: "wandb-japan/llm-leaderboard/jaster:v8"
- v 1.2.4: "wandb-japan/llm-leaderboard/jaster:v9"
- v 1.2.5: "wandb-japan/llm-leaderboard/jaster:v10"
- v 1.2.6 (latest): "wandb-japan/llm-leaderboard/jaster:v11"

Below, an example of the process of registering data in wandb's Artifacts is described for reference 

1. create dataset by following an instruction of [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)

2. register to wandb artifacts
```bash
python3 scripts/upload_jaster.py -e <wandb/entity> -p <wandb/project> -d <dataset folder> -v <version>
```

### preparation for mtbench
If you use wandb's Artifacts, this process is not necessary. The following data is currently registered in wandb's Artifacts.
If you create questions or prompts originally, you also need to create reference answers. The method for creating reference answers can be referenced from the [FastChat Readme](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

The following Japanese dataset is based on [Stability-AI/FastChat/jp-stable](https://github.com/Stability-AI/FastChat/tree/jp-stable)
- Japanese questions
  - [Stability-AI/FastChat (5d4f13a) v1.0](https://github.com/lm-sys/FastChat/commit/5d4f13a4731388ffe1453c459c357d863d87037a): 'wandb-japan/llm-leaderboard/mtbench_ja_question:v0'
  - [Stability-AI/FastChat (97d0f08) v1.1](https://github.com/Stability-AI/FastChat/commit/97d0f0863c5ee8610f00c94a293418a4209c52dd) : 'wandb-japan/llm-leaderboard/mtbench_ja_question:v1'
  - [Stability-AI/FastChat (9f220b6) v1.2 (latest)](https://github.com/lm-sys/FastChat/commit/9f220b6019eef85853237952fd2f504ac3419b72) : 'wandb-japan/llm-leaderboard/mtbench_ja_question:v3'
- Japanese prompt
  - [Stability-AI/FastChat (5d4f13a) (latest)](https://github.com/Stability-AI/FastChat/tree/jp-stable) : 'wandb-japan/llm-leaderboard/mtbench_ja_prompt:v1'
- Japanese reference answer
  - [Stability-AI/FastChat (5d4f13a)](https://github.com/Stability-AI/FastChat/tree/jp-stable) : 'wandb-japan/llm-leaderboard/mtbench_ja_referenceanswer:v0'
  - [Stability-AI/FastChat (77a69ed) (latest)]() : 'wandb-japan/llm-leaderboard/mtbench_ja_referenceanswer:v1'

The following English dataset is based on [lm-sys/FastChat/main](https://github.com/lm-sys/FastChat/tree/main)
- English questions
  - [lm-sys/FastChat (b494d0c) v1.0 (latest)](https://github.com/lm-sys/FastChat/commit/b494d0c6b4e7935f1764f8439e75da3e66beccc7) : 'wandb-japan/llm-leaderboard/mtbench_en_question:v0'
- English prompt
  - [lm-sys/FastChat (7ad1d63) (latest)](https://github.com/lm-sys/FastChat/commit/7ad1d6386288ba1a7862c11feb673425713eea5b) : 'wandb-japan/llm-leaderboard/mtbench_en_prompt:v0'
- English reference answer
  - [lm-sys/FastChat (7ad1d63)](https://github.com/lm-sys/FastChat/commit/7ad1d6386288ba1a7862c11feb673425713eea5b)
  - [lm-sys/FastChat (b494d0c) (latest)](https://github.com/lm-sys/FastChat/commit/b494d0c6b4e7935f1764f8439e75da3e66beccc7) : 'wandb-japan/llm-leaderboard/mtbench_en_referenceanswer:v0'

The following English dataset is based on [lm-sys/FastChat/main](https://github.com/lm-sys/FastChat/tree/main)
- English questions
  - [lm-sys/FastChat (b494d0c) v1.0 (latest)](https://github.com/lm-sys/FastChat/commit/b494d0c6b4e7935f1764f8439e75da3e66beccc7) : 'wandb-japan/llm-leaderboard/mtbench_en_question:v0'
- English prompt
  - [lm-sys/FastChat (7ad1d63) (latest)](https://github.com/lm-sys/FastChat/commit/7ad1d6386288ba1a7862c11feb673425713eea5b) : 'wandb-japan/llm-leaderboard/mtbench_en_prompt:v0'
- English reference answer
  - [lm-sys/FastChat (7ad1d63)](https://github.com/lm-sys/FastChat/commit/7ad1d6386288ba1a7862c11feb673425713eea5b)
  - [lm-sys/FastChat (b494d0c) (latest)](https://github.com/lm-sys/FastChat/commit/b494d0c6b4e7935f1764f8439e75da3e66beccc7) : 'wandb-japan/llm-leaderboard/mtbench_en_referenceanswer:v0'

## Create config.yaml file
1. create configs/config.yaml
```bash
cp configs/config_template.yaml configs/config.yaml
```
2. set each variable properly by following the below instruction

general
- `testmode`: The default is false. If set to true, it allows for a lightweight implementation where only 1 or 2 questions are extracted from each category. Please set it to true when you want to perform a functionality check.
- `model_name`: Model name. This is for record, so doesn't affect evaluation performance.  
- `wandb`: Information used for W&B support.
  - `entity`: Name of the W&B Entity.
  - `project`: Name of the W&B Project.
  - `run_name`: Name of the W&B run. If you set "model name" as run name, you can see easily find run on Wandb dashboard.
- `run_llm_jp_eval_ja_0_shot`, `run_llm_jp_eval_ja_few_shots`, `run_llm_jp_eval_en_0_shot`, `run_llm_jp_eval_en_few_shots`, `run_mt_bench_ja`, `run_mt_bench_en`: The default is true. If set to false, the evaluation task is skipped.
- `model`: Information used for loading model and tokenizer.
  - `api`:  If you don't use api, please set "api" as "false". If you use api, please select from "openai", "anthoropic", "google", "cohere"
  - `use_wandb_artifacts`: If you user wandb artifacts, please set true.
  - `artifacts_path`: If you user wandb artifacts, please paste the link. if not, please leave it as "".
  - `pretrained_model_name_or_path`: Name of your model. if you use openai api, put the name of model
  - `device_map`: device map. The default is "auto"
  - `load_in_8bit`: 8 bit quantization. The default is false
  - `load_in_4bit`: 4 bit quantization.The default is false
- `llm-jp-eval`: variables for llm-jp-eval
  - `max_seq_length`: The maximum length of the input. The default is 2048.
  - `target_dataset`: The dataset to evaluate. The default is all, which evaluates all datasets. Specify the dataset name (like jnli) to evaluate a specific dataset.
  - `ja_num_shots`: If run_llm_jp_eval_ja_few_shots is true, please set the num of few shots. Default is 4.
  - `en_num_shots`: If run_llm_jp_eval_en_few_shots is true, please set the num of few shots. Default is 4.
  - `torch_dtype`: Settings for fp16, bf16, fp32. The default is bf16.
  - `dataset_artifact`: URL of wandb Artifacts of evaluation dataset. Choose the version from the Data Preparation section
  - `dataset_dir`: location of the evaluation data after downloading wandb Artifacts
  - `ja`, `en`: Prompt settings for each language.
    - `custom_prompt_template`: Specification of custom prompts. The default is null. (The default prompt is using the alpaca format.)
    - `custom_fewshots_template`:  Specification of custom prompts for few-shot settings. The default is null. (The default prompt is using the alpaca format.)

- `mtbench`: variables for mtbench
  - `model_id`: ID of model. Need not to be set if you run for the first time.
  - `max_new_token`: The maximum length of the input. The default is 1024.
  - `num_gpus_per_model`: Number of GPUs per model. If you use multiple gpu, change here. The default is 1.
  - `num_gpus_total`: Number of Total GPUs. If you use multiple gpu, change here. The default is 1.
  - `max_gpu_memory`: If you specifiy the max of GPU memory, change here. The default is null.
  - `dtype`: Data type. Choose from None or float32 or float16 or bfloat16
  - `use_azure`: if you use azure openai service for evaluation, set true. If you set true, you will use openai api.
  - `custom_conv_template`: If the model is not compatible FastChat, you need to use custom conv template, and set this variable true. Then, the custom conv template you set with the following variables will be used. The defalt is false.
  - `conv_name`: Name of prompt template. The default is "custom".
  - `conv_sep`, `conv_stop_token_ids`, `conv_stop_str`, `conv_role_only_separator`: Settings for conversation.
  - `ja`, `en`: Prompt settings for language.
    - `conv_system_message`: System prompt for model.
    - `conv_roles`: Roles for conversation.
  - `dateset`: 
    - `ja`, `en`: Dataset settings for each language.
      - `question_artifacts_path`: URL of wandb Artifacts of evaluation dataset. Choose the version from the Data Preparation section
      - `test_question_artifacts_path`: URL of wandb Artifacts of evaluation dataset for testmode. Choose the version from the Data Preparation section
      - `referenceanswer_artifacts_path`: URL of wandb Artifacts of reference answer. Choose the version from the Data Preparation section
      - `test_referenceanswer_artifacts_path`: URL of wandb Artifacts of reference answer for testmode. Choose the version from the Data Preparation section
      - `judge_prompt_artifacts_path`: URL of wandb Artifacts of judge prompt. Choose the version from the Data Preparation section
      - `bench_name`: If you evaluate japanese dataset, set 'japanese_mt_bench'. If you evaluate English dataset, set 'mt_bench'.
- `github_version`: For recording. Not need to be changed

> **The generator setting is defined in the codes below.**
> - https://github.com/llm-jp/llm-jp-eval/blob/main/src/llm_jp_eval/evaluator.py
> - https://github.com/wandb/FastChat/blob/g-eval/fastchat/llm_judge/common.py

## Evaluation execution
1. run scripts/run_eval.py
```bash
python3 scripts/run_eval.py
```
2. check the wandb dashboard

## Blend runs
You can inherit some of the evaluations from past runs.

### Create config.yaml file
1. create blend_run_configs/config.yaml
```bash
cp blend_run_configs/config_template.yaml blend_run_configs/config.yaml
```
2. set each variable properly by following the below instruction

- `run_chain`: If you want to reuse past evaluation results in a new run, please set it to true.
- `new_run`: This setting is for blending runs without running new evaluations. If run_chain is set to true, this setting is disabled.
  - `entity`: Name of the W&B Entity.
  - `project`: Name of the W&B Project.
  - `run_name`: Name of the W&B run.
- `old_runs`: Please specify the tasks you want to take over from past runs. Multiple runs are permissible.
  - `run_path`: Path of the old run. The run_path is in the following format: "entity/project/run_id". It can be copied from the overview of the run.
  - `tasks`: The list of tasks to take over. Please comment out tasks that do not need to be taken over.

## Evaluation execution
This is how to integrate past evaluations into a new evaluation.
1. run scripts/run_eval.py
```bash
python3 scripts/run_eval.py
```
2. check the wandb dashboard

## Blend runs
You can also create a blended run by combining the evaluation results of multiple runs without conducting a new evaluation.
1. run scripts/blend_run.py
```bash
python3 scripts/blend_run.py
```
2. check the wandb dashboard