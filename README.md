# Nejumi Leaderboard 4
## Overview

This repository is for the Nejumi Leaderboard 4, a comprehensive evaluation platform for large language models. The leaderboard assesses both general language capabilities and alignment aspects. For detailed information about the leaderboard, please visit [Nejumi Leaderboard](https://wandb.ai/wandb-japan/llm-leaderboard4/reports/Nejumi-LLM-4--Vmlldzo5NTI0MDI0) website.

## Evaluation Metrics
Our evaluation framework incorporates a diverse set of metrics to provide a holistic assessment of model performance.

| Main Category | Category | Subcategory | Benchmarks | Details | Weight (within GLP) |
|---------------|----------|-------------|------------|---------|----------------------|
| General Language Performance (GLP) | Applied Language Skills | Expression | MT-bench (roleplay, writing, humanities) | Roleplay, writing, humanities | 1 |
| ^ | Applied Language Skills | Translation | Jaster (ALT e-to-j, ALT j-to-e) | JA↔EN translation (averaged over 0-shot and few-shot) | 1 |
| ^ | Applied Language Skills | Information Retrieval (QA) | Jaster (JSQuAD) | Japanese QA (averaged over 0-shot and few-shot) | 1 |
| ^ | Reasoning | Abstract Reasoning | ARC-AGI | Abstract pattern recognition (arc-agi-1 / arc-agi-2) | 2 |
| ^ | Reasoning | Logical Reasoning | MT-bench (reasoning) | Logical reasoning tasks | 2 |
| ^ | Reasoning | Mathematical Reasoning | Jaster (MAWPS, MGSM), MT-bench (math) | Math word problems, mathematical reasoning | 2 |
| ^ | Knowledge & QA | General Knowledge | Jaster (JCommonsenseQA, JEMHopQA, NIILC, AIO), MT-bench (STEM) | Commonsense reasoning, multi-hop QA, basic STEM knowledge | 2 |
| ^ | Knowledge & QA | Specialized Knowledge | Jaster (JMMLU, MMLU_Prox_JA), HLE | Domain knowledge evaluation (incl. PhD-level: medicine, law, engineering) | 2 |
| ^ | Foundational Language Skills | Semantic Analysis | Jaster (JNLI, JaNLI, JSeM, JSICK, JAMP) | Natural language inference, semantic similarity | 1 |
| ^ | Foundational Language Skills | Syntactic Analysis | Jaster (JCoLA-in-domain, JCoLA-out-of-domain, JBLiMP) | Grammatical acceptability | 1 |
| ^ | Application Development | Coding | SWE-Bench, JHumanEval, MT-bench (coding) | Practical coding ability, code generation | 2 |
| ^ | Application Development | Function Calling | BFCL | Function calling accuracy (single/multi-turn, irrelevant detection) | 2 |
| Alignment (ALT) | Controllability | Controllability | Jaster Control, M-IFEVAL | Instruction following, constraint adherence | 1 |
| ^ | Ethics & Morality | Ethics & Morality | Jaster (CommonsenseMoralityJA) | Ethical and moral judgement | 1 |
| ^ | Safety | Toxicity | Toxicity | Fairness, social norms, prohibited behavior, violation categories | 1 |
| ^ | Bias | Bias | JBBQ | Japanese bias benchmark (1 - avg_abs_bias_score) | 1 |
| ^ | Truthfulness | Truthfulness | JTruthfulQA, HalluLens | Factuality assessment, hallucination suppression (refusal rate) | 1 |
| ^ | Robustness | Robustness | JMMLU Robust | Consistency and robustness under varied question formats | 1 |


- metrics with (0, 2-shot) are averaged across both settings.
- Metrics marked with an asterisk (*) evaluate control capabilities.
- For MT-bench, [StabilityAI's MT-Bench JP](https://github.com/Stability-AI/FastChat/tree/jp-stable) is used with GPT-4o-2024-05-13 as the model to evaluate.
- vLLM is leveraged for efficient inference.
- **Alignment data may contain sensitive information and the default setting does not include it in this repository. If you want to evaluate your models agains Alinghment data, please check each dataset instruction carefully**

## Implementation Guide

### Environment Setup
1. Clone the repository
```bash
git clone https://github.com/wandb/llm-leaderboard.git
cd llm-leaderboard
```

2. Create a `.env` file in the repository root
```
WANDB_API_KEY=
OPENAI_API_KEY=
LANG=ja_JP.UTF-8
# OpenAI compatible (e.g. OpenRouter / vLLM Gateway)
OPENAI_COMPATIBLE_API_KEY=

# Azure OpenAI (use if you evaluate via Azure)
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
OPENAI_API_TYPE=azure

# Other API keys (set as needed)
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
COHERE_API_KEY=
MISTRAL_API_KEY=
UPSTAGE_API_KEY=
XAI_API_KEY=
DEEPSEEK_API_KEY=

# Hugging Face
HUGGINGFACE_HUB_TOKEN=
HF_TOKEN=${HUGGINGFACE_HUB_TOKEN}

# (Optional) AWS for Bedrock family
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=

# (Optional) SWE-bench server key
SWE_API_KEY=
```

3. (First time only) Create Docker network
```bash
docker network create llm-stack-network || true
```

### Dataset Preparation

For detailed instructions on dataset preparation and caveate, please refer to [scripts/data_uploader/README.md](./scripts/data_uploader/README.md).

In Nejumi Leadeboard4, the following dataset are used.

**Please ensure to thoroughly review the terms of use for each dataset before using them.**

1. [jaster](https://github.com/llm-jp/llm-jp-eval/tree/nejumi3-data)(Apache-2.0 license)
2. [MT-Bench-JA](https://github.com/Stability-AI/FastChat/tree/jp-stable) (Apache-2.0 license)
3. [JBBQ](https://github.com/ynklab/JBBQ_data?tab=readme-ov-file) (Creative Commons Attribution 4.0 International License.)
4. LINE Yahoo Inappropriate Speech Evaluation Dataset (not publically available)
5. [JTruthfulQA](https://github.com/nlp-waseda/JTruthfulQA) (Creative Commons Attribution 4.0 International License.)
6. [SWE-bench](https://www.swebench.com/) (Apache 2.0 license)
7. [BFCL](https://github.com/salesforce/CoT-Benmarking-Tool) (BSD 3-Clause "New" or "Revised" License)
8. [HalluLens](https://github.com/idea-research/HalluLens) (MIT license)
9. [HLE-JA](https://huggingface.co/datasets/Hitachi-AIN/HLE-JA) (Apache 2.0 license)
10. [ARC-AGI-2](https://github.com/google-deepmind/arc-agi) (CC-BY-SA-4.0 license)
11. [M-IFEval](https://github.com/google-deepmind/instruction-following-eval) (Apache 2.0 license)



### Configuration

#### Base configuration

The `configs/base_config.yaml` file contains the shared settings. Create per-model YAMLs under `configs/` that override only the necessary parts. Below are key sections used in this repository (names and placement reflect the actual file structure):

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `entity`: Name of the W&B Entity.
    - `project`: Name of the W&B Project.
    - `run_name`: Name of the W&B run. Please set up run name in a model-specific config.
- **testmode:** Default is false. Set to true for lightweight implementation with a small number of questions per category (for functionality checks).
- **inference_interval:** Set inference interval in seconds. This is particularly effective when there are rate limits, such as with APIs.
- **run:** Set to true for each evaluation dataset you want to run.
- **model:** Model metadata (kept minimal in base; overrides go in each model config).
    - `artifacts_path`: Path of the W&B artifact if loading models from artifacts (optional).

- **vllm:** vLLM launch parameters (only used for open‑weights via vLLM).
    - `vllm_tag`: Docker image tag for vLLM (default: latest).
    - `disable_triton_mma`: Workaround for older GPUs.
    - `lifecycle`: Container lifecycle policy.
    - `dtype`, `max_model_len`, `device_map`, `gpu_memory_utilization`, `trust_remote_code`, `chat_template`, `extra_args`: vLLM-specific options.

- **generator:** Settings for generation. For more details, refer to the [generation_utils](https://huggingface.co/docs/transformers/internal/generation_utils) in Hugging Face Transformers.
    - `top_p`: top-p sampling. Default is 1.0.
    - `temperature`: The temperature for sampling. Default is 0.1.
    - `max_tokens`: Maximum number of tokens to generate. This value will be overwritten in the script.

- **num_few_shots:**  Number of few-shot examples to use.

- **github_version:** For recording, not required to be changed.

- **jaster:**  Settings for the Jaster dataset.
    - `artifacts_path`: URL of the WandB Artifact for the Jaster dataset.
    - `dataset_dir`: Directory of the Jaster dataset after downloading the Artifact.
    - `jhumaneval`: Settings for the jhumaneval dataset.
        - `dify_sandbox`: Dify Sandbox configuration for secure code execution.
            - `endpoint`: Sandbox service endpoint.
            - `api_key`: API key for authentication.

- **jmmlu_robustness:** Whether to include the JMMLU Robustness evaluation. Default is True.

- **jbbq:** Settings for the JBBQ dataset.
    - `artifacts_path`: URL of the WandB Artifact for the JBBQ dataset.
    - `dataset_dir`: Directory of the JBBQ dataset after downloading the Artifact.

- **toxicity:** Settings for the toxicity evaluation.
    - `artifact_path`: URL of the WandB Artifact of the toxicity dataset.
    - `judge_prompts_path`: URL of the WandB Artifact of the toxicity judge prompts.
    - `max_workers`: Number of workers for parallel processing.
    - `judge_model`: Model used for toxicity judgment. Default is `gpt-4o-2024-05-13`

- **jtruthfulqa:** Settings for the JTruthfulQA dataset.
    - `artifact_path`: URL of the WandB Artifact for the JTruthfulQA dataset.
    - `roberta_model_name`: Name of the RoBERTa model used for evaluation. Default is 'nlp-waseda/roberta_jtruthfulqa'.

- **swebench:** Settings for the SWE-bench dataset (matches `configs/base_config.yaml`).
    - `artifacts_path`: URL of the WandB Artifact for the SWE-bench dataset.
    - `dataset_dir`: Directory of the SWE-bench dataset after downloading the Artifact.
    - `max_samples`: Number of samples to use for evaluation.
    - `max_tokens`: Maximum number of tokens to generate.
    - `max_workers`: Number of workers for parallel processing.
    - `evaluation_method`: Choose 'official' or 'docker'.
    - `fc_enabled`: Enable function calling to enforce unified diff output.
    - `api_server`: Remote evaluation settings (`enabled`, `endpoint`, `api_key`, `timeout_sec`).

- **mtbench:** Settings for the MT-Bench evaluation.
    - `temperature_override`: Override the temperature for each category of the MT-Bench.
    - `question_artifacts_path`: URL of the WandB Artifact for the MT-Bench questions.
    - `referenceanswer_artifacts_path`: URL of the WandB Artifact for the MT-Bench reference answers.
    - `judge_prompt_artifacts_path`: URL of the WandB Artifact for the MT-Bench judge prompts.
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

- **bfcl:** Settings for the BFCL dataset.
    - `num_threads`: Number of parallel threads to use
    - `temperature`: Temperature
    - `artifacts_path`: URL of the WandB Artifact for the BFCL dataset.
    - For details, please check scripts/evaluator/evaluate_utils/bfcl_pkg/README.md. Added supplemental explanation at the end in Japanese

- **hallulens:** Settings for the HalluLens dataset.
    - `artifacts_path`: URL of the WandB Artifact for the HalluLens dataset.
    - `judge_model`: Model used for judging the generated responses.

- **hle:** Settings for the HLE-JA dataset.
    - `artifact_path`: URL of the WandB Artifact for the HLE-JA dataset.
    - `judge_model`: Model used for judging the generated responses.

- **arc_agi:** Settings for the ARC-AGI dataset.
    - `arc_agi_1_artifacts_path`: URL of the WandB Artifact for the ARC-AGI-1 dataset.
    - `arc_agi_2_artifacts_path`: URL of the WandB Artifact for the ARC-AGI-2 dataset.

- **m_ifeval:** Settings for the M-IFEval dataset.
    - `artifacts_path`: URL of the WandB Artifact for the M-IFEval dataset.


### Model configuration
After setting up the base-configuration file, the next step is to set up a configuration file for model under `configs/`.
#### API Model Configurations
This framework supports evaluating models using APIs such as OpenAI, Anthropic, Google, and Cohere. You need to create a separate config file for each API model. For example, the config file for OpenAI's gpt-4o-2024-05-13 would be named `configs/config-gpt-4o-2024-05-13.yaml`.

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `run_name`: Name of the W&B run.
- **api:** Choose the API to use from `openai`, `anthropic`, `google`, `amazon_bedrock`.
- **batch_size:** Batch size for API calls (recommended: 32).
- **model:** Information about the API model. 
    - `pretrained_model_name_or_path`: API model name (e.g., `gpt-5-2025-08-07`).
    - `size_category`: Use `api`.
    - `size`: Leave as null for API models.
    - `release_date`: Model release date (MM/DD/YYYY).
    - `bfcl_model_id`: Select an ID from `scripts/evaluator/evaluate_utils/bfcl_pkg/SUPPORTED_MODELS.md` or use a common handler (e.g., `OpenAIResponsesHandler-FC`).

#### Other Model Configurations

This framework also supports evaluating models using VLLM. You need to create a separate config file for each VLLM model. For example, the config file for Microsoft's Phi-3-medium-128k-instruct would be named `configs/config-Phi-3-medium-128k-instruct.yaml`.

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `run_name`: Name of the W&B run.
- **api:** Set to `vllm` to indicate using a VLLM model.
- **num_gpus:** Number of GPUs to use.
- **batch_size:** Batch size for VLLM (recommended: 256).
- **model:** Information about the model.
    - `artifacts_path`: When loading a model from wandb artifacts, it is necessary to include a description. If not, there is no need to write it. Example notation: wandb-japan/llm-leaderboard/llm-jp-13b-instruct-lora-jaster-v1.0:v0   
    - `pretrained_model_name_or_path`: Name of the VLLM model.
    - `bfcl_model_name`: Please select from scripts/evaluator/evaluate_utils/bfcl_pkg/SUPPORTED_MODELS.md. Since adding new models can be complex, Nejumi Leaderboard provides an `oss_handler` for easier integration. If it is difficult to create a dedicated handler for each model, please use the `oss_handler` as a default option.
    - `chat_template`: Path to the chat template file (if needed).
    - `size_category`: Specify model size category. In Nejumi Leaderboard, the category is defined as "10B<", "10B<= <30B", "<=30B" and "api".
    - `size`: Model size (parameter).
    - `release_date`: Model release date (MM/DD/YYYY).
    - `max_model_len`: Maximum token length of the input (if needed).

#### VLLM Configuration
When using vLLM models, you can add additional vLLM-specific configurations:

```yaml
vllm:
  vllm_tag: v0.5.5  # Docker image tag for vLLM
  disable_triton_mma: true  # Set to true if you encounter Triton MMA errors
  lifecycle: stop_restart  # vLLM container lifecycle: 'stop_restart' (default) or 'always_on'
  dtype: half  # Data type for model inference
  extra_args:  # Additional arguments to pass to vLLM server
    - --enable-lora
    - --trust-remote-code
```

#### Create Chat template (needed for models except for API)
1. create chat_templates/model_id.jinja
If the chat_template is specified in the tokenizer_config.json of the evaluation model, create a .jinja file with that configuration.
If chat_template is not specified in tokenizer_config.json, refer to the model card or other relevant documentation to create a chat_template and document it in a .jinja file.

2. test chat_templates
If you want to check the output of the chat_templates, you can use the following script:
```bash
python3 scripts/test_chat_template.py -m <model_id> -c <chat_template>
```
If the model ID and chat_template are the same, you can omit -c <chat_template>.


## Evaluation Execution
Once you prepare the dataset and the configuration files, you can run the evaluation process.

### Using Docker Compose

Basic flow (requires `.env` and a model YAML under `configs/`):

```bash
# 1) Create the network once (safe if it already exists)
docker network create llm-stack-network || true

# 2) Run the evaluation (example: OpenAI o3 config)
bash ./run_with_compose.sh config-o3-2025-04-16.yaml

# Add -d to see more debug output
# bash ./run_with_compose.sh config-o3-2025-04-16.yaml -d
```

The script does the following automatically:
- Starts **ssrf-proxy** and **dify-sandbox**
- If the target is a **vLLM** model, starts the vLLM container and waits for readiness
- Runs `scripts/run_eval.py -c <your YAML>` inside the evaluation container

### Manual Execution (Advanced / Optional)

Use only if you want to run directly without `run_with_compose.sh`.
- **-c (config):** `python3 scripts/run_eval.py -c config-gpt-4o-2024-05-13.yaml`
- **-s (select-config):** `python3 scripts/run_eval.py -s`

### SWE-Bench and API Server Docs

For SWE‑Bench evaluation details and the remote evaluation API server, see:

- `docs/README_swebench_ja.md` (dataset creation, prompts, patch rules)
- `scripts/evaluator/evaluate_utils/swebench_pkg/swebench_api_server.md` (API server runbook)

### Troubleshooting

#### Container Name Conflicts
If you encounter "container name already in use" errors:
```bash
docker rm -f llm-stack-vllm-1 llm-leaderboard
```

#### vLLM Memory Issues
For models requiring large GPU memory:
- Adjust `max_model_len` in your config file
- Use smaller `batch_size`
- Enable quantization if supported

#### BFCL Evaluation Issues
If BFCL evaluation fails with model name mismatches:
- Check if your model requires the "-FC" suffix
- Add `bfcl_model_name` to your config if the BFCL model name differs from the base model

The results of the evaluation will be logged to the specified W&B project.

## When you want to edit runs or add additional evaluation metrics
Refer to the in-repo docs above or open an issue/PR. The previously referenced `blend_run_configs` guide is not used in this repository.



## Contributing
Contributions to this repository is welcom. Please submit your suggestions via pull requests. Please note that we may not accept all pull requests.

## License
This repository is available for commercial use. However, please adhere to the respective rights and licenses of each evaluation dataset used.

## Contact
For questions or support, please concatct to contact-jp@wandb.com.