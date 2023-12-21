# Nejumi-leaderboard Neo


## Set up
1. Set up environment variables
```
export WANDB_API_KEY=<your WANDB_API_KEY>
# if needed, please set the following API KEY too
export OPENAI_API_KEY=<your OPENAI_API_KEY>
export ANTHROPIC_API_KEY=<your ANTHROPIC_API_KEY>
```

## Data Prepartion 
### preparation for llm-jp-eval
If you use wandb's Artifacts, this process is not necessary. The following data is currently registered in wandb's Artifacts.

- v 1.0.0: "wandb-japan/llm-leaderboard/jaster:v0"
- v 1.1.0: "wandb-japan/llm-leaderboard/jaster:v2"

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
  - [Stability-AI/FastChat (5d4f13a)](https://github.com/Stability-AI/FastChat/tree/jp-stable) : 'wandb-japan/llm-leaderboard/mtbench_ja_prompt:v0'
- reference answer
  - [Stability-AI/FastChat (5d4f13a)](https://github.com/Stability-AI/FastChat/tree/jp-stable) : 'wandb-japan/llm-leaderboard/mtbench_ja_referenceanswer:v0'


Below, an example of the process of registering data in wandb's Artifacts is described for reference 
```bash
# register questions
  python3 scripts/upload_mtbench_question.py -e <wandb/entity> -p <wandb/project> -v <data version> -f "your path"
```

## Evaluation
Please follow the instructions below. By executing these steps, the results will be aggregated and displayed on the wandb dashboard
1. create configs/config.yaml
```bash
cp configs/config_template.yaml configs/config.yaml
```
2. run scripts/run_eval.py
```bash
python3 scripts/run_eval.py
```
3. check wandb dashboard from the generated wandb URL