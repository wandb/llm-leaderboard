# Nejumi-leaderboard Neo


## 目次

- [Install](#Install)
- [Data Preparation](#Data Preparation)


## Install
- [ ] 複数のrepositoryがある場合のinstal方法を考える必要がある
- [ ] サブモジュールがあるrepositoryのinstall instructionを書く

## Set up
1. Set up environment variables
```
export WANDB_API_KEY=<your WANDB_API_KEY>
export OPENAI_API_KEY=<your OPENAI_API_KEY>
```

## Data Prepartion 
### preparation for llm-jp-eval
If you use wandb's Artifacts, this process is not necessary. The following data is currently registered in wandb's Artifacts.

- v 1.0.0: "wandb-japan/llm-leaderboard/jaster:v0"
- v 1.1.0: "wandb-japan/llm-leaderboard/jaster:v1"

Below, an example of the process of registering data in wandb's Artifacts is described for reference 
0. library install
```bash
cd llm-jp-eval
poetry install
```

1. create jaster dataset
```bash
  poetry run python scripts/preprocess_dataset.py  \
  --dataset-name all  \
  --output-dir dataset \
```
2. register to wandb artifacts
```bash
python3 scripts/upload_jaster.py -e wandb-japan -p llm-leaderboard -d llm-jp-eval/dataset -v 1.0.0
```

### preparation for mtbench
If you use wandb's Artifacts, this process is not necessary. The following data is currently registered in wandb's Artifacts.
If you create questions or prompts originally, you also need to create reference answers. The method for creating reference answers can be referenced from the [FastChat Readme](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

- japanese questions
  - 5d4f13a of [Stability-AI/FastChat](https://github.com/Stability-AI/FastChat/tree/jp-stable) : "wandb-japan/llm-leaderboard/jaster:v0"
- japanese prompt
  - 5d4f13a of [Stability-AI/FastChat](https://github.com/Stability-AI/FastChat/tree/jp-stable) : "wandb-japan/llm-leaderboard/jaster:v0"
- reference answer
  - 5d4f13a of [Stability-AI/FastChat](https://github.com/Stability-AI/FastChat/tree/jp-stable) : "wandb-japan/llm-leaderboard/jaster:v0"



Below, an example of the process of registering data in wandb's Artifacts is described for reference 
1. register questions
```bash
  python3 scripts/upload_mtbench_question.py -e wandb-japan -p llm-leaderboard -v 20231130
```

2. register judge prompts


3. register reference answer


## Evaluation
Please follow the instructions below. By executing these steps, the results will be aggregated and displayed on the wandb dashboard
1. update configs/config.yaml
[ ] detailed explanation of config file is needed
2. run scripts/run_eval.py
```bash
python3 scripts/run_eval.py
```
3. check wandb dashboard from the generated wandb URL