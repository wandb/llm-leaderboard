# Nejumi-leaderboard3

## Evaluation Framework for Japanese Language Models

This repository provides a framework for evaluating Japanese Language Models (JLMs) on various tasks, including foundational language capabilities, alignment capabilities, and translation. 

## License
### Not for Commercial Use
The contents of this repository are not permitted for commercial use.

## Set up
1. Set up environment variables
```
export WANDB_API_KEY=<your WANDB_API_KEY>
export OPENAI_API_KEY=<your OPENAI_API_KEY>
export LANG=ja_JP.UTF-8
# if needed, set the following API KEY too
export ANTHROPIC_API_KEY=<your ANTHROPIC_API_KEY>
export GOOGLE_API_KEY=<your GOOGLE_API_KEY>
export COHERE_API_KEY=<your COHERE_API_KEY>
export MISTRAL_API_KEY=<your MISTRAL_API_KEY>
export AWS_ACCESS_KEY_ID=<your AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<your AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=<your AWS_DEFAULT_REGION>
export UPSTAGE_API_KEY=<your UPSTAGE_API_KEY>
# if needed, please login in huggingface
huggingface-cli login
```

2. Clone the repository
```bash
git clone https://github.com/wandb/llm-leaderboard.git
cd llm-leaderboard
```

3. Set up the Python environment
```bash
pip install -r requirements.txt
```

## Data Prepartion 
### preparation for llm-jp-eval
If you use wandb's Artifacts, this process is not necessary. The following data is currently registered in wandb's Artifacts.

- v 1.3.0 (latest): "wandb-japan/llm-leaderboard3/jaster:v6"

Below, an example of the process of registering data in wandb's Artifacts is described for reference 

1. create dataset by following an instruction of [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval/tree/wandb-nejumi2)

2. register to wandb artifacts
```bash
python3 scripts/upload_jaster.py -e <wandb/entity> -p <wandb/project> -d <dataset folder> -v <version>
```

### preparation for mtbench
If you use wandb's Artifacts, this process is not necessary. The following data is currently registered in wandb's Artifacts.
If you create questions or prompts originally, you also need to create reference answers. The method for creating reference answers can be referenced from the [FastChat Readme](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

The following data are based on [Stability-AI/FastChat/jp-stable](https://github.com/Stability-AI/FastChat/tree/jp-stable)
- japanese questions
  - Stability-AI/FastChat (5d4f13a) v1.0 : 'wandb-japan/llm-leaderboard/mtbench_ja_question:v0'
  - [Stability-AI/FastChat (97d0f08) v1.1](https://github.com/Stability-AI/FastChat/commit/97d0f0863c5ee8610f00c94a293418a4209c52dd) : 'wandb-japan/llm-leaderboard/mtbench_ja_question:v1'
  - [wandb/llm-leaderboard3 (8208d2a) (latest)](https://github.com/wandb/llm-leaderboard/commit/8208d2a2f9ae5b7f264b3d3cd4f28334afb7af13) : 'wandb-japan/llm-leaderboard3/mtbench_ja_question:v1'
- japanese prompt
  - [Stability-AI/FastChat (5d4f13a) (latest)](https://github.com/Stability-AI/FastChat/tree/jp-stable) : 'wandb-japan/llm-leaderboard3/mtbench_ja_prompt:v1'
- reference answer
  - [Stability-AI/FastChat (5d4f13a)](https://github.com/Stability-AI/FastChat/tree/jp-stable) : 'wandb-japan/llm-leaderboard3/mtbench_ja_referenceanswer:v1'
  - [wandb/llm-leaderboard (8208d2a) (latest)](https://github.com/wandb/llm-leaderboard/commit/8208d2a2f9ae5b7f264b3d3cd4f28334afb7af13) : 'wandb-japan/llm-leaderboard3/mtbench_ja_referenceanswer:v1'


Below, an example of the process of registering data in wandb's Artifacts is described for reference 
```bash
# register questions
  python3 scripts/upload_mtbench_question.py -e <wandb/entity> -p <wandb/project> -v <data version> -f "your path"
```

## Create chat_template file
1. create chat_templates/model_id.jinja
If the chat_template is specified in the tokenizer_config.json of the evaluation model, create a .jinja file with that configuration.
If chat_template is not specified in tokenizer_config.json, refer to the model card or other relevant documentation to create a chat_template and document it in a .jinja file.

2. test chat_templates
If you want to check the output of the chat_templates, you can use the following script:
```bash
python3 scripts/test_chat_template.py -m <model_id> -c <chat_template>
```
If the model ID and chat_template are the same, you can omit -c <chat_template>.

## Configuration

The `base_config.yaml` file contains basic settings, and you can create a separate YAML file for model-specific settings. This allows for easy customization of settings for each model while maintaining a consistent base configuration.

### General Settings

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `entity`: Name of the W&B Entity.
    - `project`: Name of the W&B Project.
    - `run_name`: Name of the W&B run. Please set up run name in a model-specific config.
- **github_version:** For recording, not required to be changed.
- **testmode:** Default is false. Set to true for lightweight implementation with a small number of questions per category (for functionality checks).
- **inference_interval:** Set inference interval in seconds
- **run:** Set to true for each evaluation dataset you want to run.
    - `jaster`: True for evaluating foundational language capabilities.
    - `ALT`: True for evaluating alignment capabilities. This option is not available to general users as it includes private datasets.
    
### Model Settings

- **model:** Information about the model.
    - `use_wandb_artifacts`: Whether to use WandB artifacts for the model.
    - `max_model_len`: Maximum token length of the input.
    - `chat_template`: Path to the chat template file. This is required for open-weights models.
    - `dtype`: Data type. Choose from float32, float16, bfloat16.
    - `trust_remote_code`:  Default is true.
    - `device_map`: Device map. Default is "auto".
    - `load_in_8bit`: 8-bit quantization. Default is false.
    - `load_in_4bit`: 4-bit quantization. Default is false.

- **generator:** Settings for generation. For more details, refer to the [generation_utils](https://huggingface.co/docs/transformers/internal/generation_utils) in Hugging Face Transformers.
    - `top_p`: top-p sampling. Default is 1.0.
    - `temperature`: The temperature for sampling. Default is 0.1.
    - `max_tokens`: Maximum number of tokens to generate. This value will be overwritten in the script.

- **num_few_shots:**  Number of few-shot examples to use.

- **jaster:**  Settings for the Jaster dataset.
    - `artifacts_path`: URL of the W&B Artifact for the Jaster dataset.
    - `dataset_dir`: Directory for the Jaster dataset after downloading the Artifact.

- **jmmlu_robustness:** Whether to include the JMMLU Robustness evaluation. Default is True.

- **lctg:** Settings for the LCTG dataset.
    - `artifacts_path`: URL of the W&B Artifact for the LCTG dataset.
    - `dataset_dir`: Directory for the LCTG dataset after downloading the Artifact.

- **jbbq:** Settings for the JBBQ dataset.
    - `artifacts_path`: URL of the W&B Artifact for the JBBQ dataset.
    - `dataset_dir`: Directory for the JBBQ dataset after downloading the Artifact.

- **toxicity:** Settings for the toxicity evaluation.
    - `artifact_path`: URL of the W&B Artifact for the toxicity dataset.
    - `judge_prompts_path`: URL of the W&B Artifact for the toxicity judge prompts.
    - `max_workers`: Number of workers for parallel processing.
    - `judge_model`: Model used for toxicity judgment. Default is `gpt-4o-2024-05-13`

- **mtbench:** Settings for the MT-Bench evaluation.
    - `temperature_override`: Override the temperature for each category of the MT-Bench.
    - `question_artifacts_path`: URL of the W&B Artifact for the MT-Bench questions.
    - `referenceanswer_artifacts_path`: URL of the W&B Artifact for the MT-Bench reference answers.
    - `judge_prompt_artifacts_path`: URL of the W&B Artifact for the MT-Bench judge prompts.
    - `bench_name`: Choose 'japanese_mt_bench' for the Japanese MT-Bench, or 'mt_bench' for the English version.
    - `model_id`: The name of the model. You can replace this with a different value if needed.
    - `question_begin`: Starting position for the question in the generated text.
    - `question_end`: Ending position for the question in the generated text.
    - `max_new_token`: Maximum number of new tokens to generate.
    - `num_choices`: Number of choices to generate.
    - `num_gpus_per_model`: Number of GPUs to use per model.
    - `num_gpus_total`: Total number of GPUs to use.
    - `max_gpu_memory`: Maximum GPU memory to use (leave as null to use the default).
    - `dtype`: Data type. Choose from None, float32, float16, bfloat16.
    - `judge_model`: Model used for judging the generated responses. Default is `gpt-4o-2024-05-13`
    - `mode`: Mode of evaluation. Default is 'single'.
    - `baseline_model`: Model used for comparison. Leave as null for default behavior.
    - `parallel`: Number of parallel threads to use.
    - `first_n`: Number of generated responses to use for comparison. Leave as null for default behavior.


## API Model Configurations

This framework supports evaluating models using APIs such as OpenAI, Anthropic, Google, and Cohere. You need to create a separate config file for each API model. For example, the config file for OpenAI's gpt-4o-2024-05-13 would be named `configs/config-gpt-4o-2024-05-13.yaml`.

### API Model Configuration Settings

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `run_name`: Name of the W&B run.
- **api:** Choose the API to use from `openai`, `anthropic`, `google`, `amazon_bedrock`.
- **batch_size:** Batch size for API calls (recommended: 32).
- **model:** Information about the model.
    - `pretrained_model_name_or_path`: Name of the API model.
    - `size_category`: Specify "api" to indicate using an API model.
    - `size`: Model size (leave as null for API models).
    - `release_date`: Model release date. (MM/DD/YYYY)

## VLLM Model Configurations

This framework also supports evaluating models using VLLM.  You need to create a separate config file for each VLLM model. For example, the config file for Microsoft's Phi-3-medium-128k-instruct would be named `configs/config-Phi-3-medium-128k-instruct.yaml`.

### VLLM Model Configuration Settings

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `run_name`: Name of the W&B run.
- **api:** Set to `vllm` to indicate using a VLLM model.
- **num_gpus:** Number of GPUs to use.
- **batch_size:** Batch size for VLLM (recommended: 256).
- **model:** Information about the model.
    - `use_wandb_artifacts`: Set to true if you want to use wandb artifacts.
    - `pretrained_model_name_or_path`: Name of the VLLM model.
    - `chat_template`: Path to the chat template file (if needed).
    - `size_category`: Specify "api" to indicate using an API model.
    - `size`: Model size (parameter).
    - `release_date`: Model release date (MM/DD/YYYY).
    - `max_model_len`: Maximum token length of the input (if needed).

## Evaluation Execution

1. **Run the evaluation script:**

    You can use either `-c` or `-s` option:
    - **-c (config):** Specify the config file by its name, e.g., `python3 scripts/run_eval.py -c config-gpt-4o-2024-05-13.yaml`
    - **-s (select-config):** Select from a list of available config files. This option is useful if you have multiple config files. 
   ```bash
   python3 scripts/run_eval.py -s
   ```

2. **Check the W&B dashboard:** The results of the evaluation will be logged to the specified W&B project.

## Blender

This feature allows you to blend inference results later or resume a run by carrying over the previous results.

### blender Configuration Settings

- **run_chain:** Please select false to use the blend feature and true to use the resume feature.

- **num_few_shots:**  Number of few-shot examples to use.

- **model:** Information about the model. (No configuration is required to use the resume feature.)
    - `use_wandb_artifacts`: Set to true if you want to use wandb artifacts.
    - `pretrained_model_name_or_path`: Name of the VLLM model.
    - `chat_template`: Path to the chat template file (if needed).
    - `size_category`: Specify "api" to indicate using an API model.
    - `size`: Model size (parameter).
    - `release_date`: Model release date (MM/DD/YYYY).
    - `max_model_len`: Maximum token length of the input (if needed).

- **new_run:** Information used for Weights & Biases (W&B) support. (No configuration is required to use the resume feature.)
    - `entity`: Name of the W&B Entity.
    - `project`: Name of the W&B Project.
    - `run_name`: Name of the W&B run. Please set up run name in a model-specific config.

- **old_run:** Please specify the tasks you want to carry over from past runs. Multiple runs are permissible.
    -`run_path`: run path of the W&B old_run. 
     `dataset`: The list of tasks to take over. Please comment out tasks that do not need to be taken over.

### blend run

1. **Setting blend_config:**

   ```bash
   cp -ip blend_run_configs/config_template.yaml blend_run_configs/blend_config.yaml
   ```

2. **Run the blend script:**

   ```bash
   python3 scripts/blend_run.py
   ```

3. **Check the W&B dashboard:** The results of the evaluation will be logged to the specified W&B project.

### resume run

1. **Setting blend_config:**

   ```bash
   cp -ip blend_run_configs/config_template.yaml blend_run_configs/blend_config.yaml
   ```

2. **Run the evaluation script:**

    You can use either `-c` or `-s` option:
    - **-c (config):** Specify the config file by its name, e.g., `python3 scripts/run_eval.py -c config-gpt-4o-2024-05-13.yaml`
    - **-s (select-config):** Select from a list of available config files. This option is useful if you have multiple config files. 
   ```bash
   python3 scripts/run_eval.py -s
   ```

3. **Check the W&B dashboard:** The results of the evaluation will be logged to the specified W&B project.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

## License

This project is licensed under the Apache 2.0 License.
