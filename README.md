# Nejumi-leaderboard Neo

## Set up
1. Set up environment variables
```
export WANDB_API_KEY=<your WANDB_API_KEY>
# if needed, please set the following API KEY too
export OPENAI_API_KEY=<your OPENAI_API_KEY>
export ANTHROPIC_API_KEY=<your ANTHROPIC_API_KEY>
export GOOGLE_API_KEY=<your GOOGLE_API_KEY>
export COHERE_API_KEY=<your COHERE_API_KEY>
# if needed, please login in huggingface
huggingface-cli login
# if needed
export LANG=ja_JP.UTF-8
```



## Data Prepartion 
### preparation for llm-jp-eval
If you use wandb's Artifacts, this process is not necessary. The following data is currently registered in wandb's Artifacts.

- v 1.0.0: "wandb-japan/llm-leaderboard/jaster:v0"
- v 1.1.0: "wandb-japan/llm-leaderboard/jaster:v3"

Below, an example of the process of registering data in wandb's Artifacts is described for reference 

1. create dataset by following an instruction of [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval/tree/wandb-nejumi)

2. register to wandb artifacts
```bash
python3 scripts/upload_jaster.py -e <wandb/entity> -p <wandb/project> -d <dataset folder> -v <version>
```

### preparation for mtbench
If you use wandb's Artifacts, this process is not necessary. The following data is currently registered in wandb's Artifacts.
If you create questions or prompts originally, you also need to create reference answers. The method for creating reference answers can be referenced from the [FastChat Readme](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

- japanese questions
  - [Stability-AI/FastChat (5d4f13a)](https://github.com/Stability-AI/FastChat/tree/jp-stable) : 'wandb-japan/llm-leaderboard/mtbench_ja_question:v0'
- japanese prompt
  - [Stability-AI/FastChat (5d4f13a)](https://github.com/Stability-AI/FastChat/tree/jp-stable) : 'wandb-japan/llm-leaderboard/mtbench_ja_prompt:v1'
- reference answer
  - [Stability-AI/FastChat (5d4f13a)](https://github.com/Stability-AI/FastChat/tree/jp-stable) : 'wandb-japan/llm-leaderboard/mtbench_ja_referenceanswer:v0'


Below, an example of the process of registering data in wandb's Artifacts is described for reference 
```bash
# register questions
  python3 scripts/upload_mtbench_question.py -e <wandb/entity> -p <wandb/project> -v <data version> -f "your path"
```
## Create config.yaml file
1. create configs/config.yaml
```bash
cp configs/config_template.yaml configs/config.yaml
```
2. set each variable properly by following the below instruction
   
- `wandb`: Information used for W&B support.
  - `entity`: Name of the W&B Entity.
  - `project`: Name of the W&B Project.
  - `run_name`: Name of the W&B run.
- `github_version`: For recording. Not need to be changed
- `testmode`: The default is false. If set to true, it allows for a lightweight implementation where only 1 or 2 questions are extracted from each category. Please set it to true when you want to perform a functionality check
- `api`:  If you don't use api, please set "api" as "false". If you use api, please select from "openai", "anthoropic", "google", "cohere"
- model:
  `_target_`: transformers.AutoModelForCausalLM.from_pretrained
  `pretrained_model_name_or_path`: Name of your model. if you use openai api, put the name of model
  `trust_remote_code`: true
  `device_map`: device map. The default is "auto"
  `load_in_8bit`: 8 bit quantization. The default is false
  `load_in_4bit`: 4 bit quantization.The default is false
- generator: Settings for generation. For more details, refer to the [generation_utils](https://huggingface.co/docs/transformers/internal/generation_utils)  in huggingface transformers.
  - `top_p`: top-p sampling. The default is 1.0.
  - `top_k`: top-k sampling. Default is commented out.
  - `temperature`: The temperature for sampling. Default is commented out.
  - `repetition_penalty`: Repetition penalty. The default is 1.0.






   
## Evaluation execution
1. run scripts/run_eval.py
```bash
python3 scripts/run_eval.py
```
2. check the wandb dashboard
