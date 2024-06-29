# Dataset Preparation
How to prepare the data is explained here.

## JASTER 
jaster is the Japanese evaluation dataset managed by llm-jp in [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval). [llm-jp-eval/nejumi3-data](https://github.com/llm-jp/llm-jp-eval/tree/nejumi3-data) is created for data preparation for Nejumi Leaderboard3.

 #### For wandb multinant SaaS users
The artifact has been registered and the paths have been already put in the default config files.
The artifact path is in `configs/base-config.yaml`.

#### For wandb on-premises or dedicated cloud users
Artifact registration is required.

1. Download the dataset from the above artifacts by logging on wandb multinant SaaS. or create jaster dataset by following the instruction in [llm-jp-eval/nejumi3-data](https://github.com/llm-jp/llm-jp-eval/tree/nejumi3-data).
2. Upload the dataset with 
```bash
python3 scripts/data_uploader.upload_dataset.py -e <your wandb entity> -p <your wandb project> -d <pass of jaster dataset> -n jaster -v <version>
```

## MT-Bench
The data in [Stability-AI/FastChat/jp-stable](https://github.com/Stability-AI/FastChat/tree/jp-stable) are used in Nejumi Leaderboard3.

#### For wandb multinant SaaS users
The artifact has been registered and the paths have been already put in the default config files.
The artifact path is in `configs/base-config.yaml`

#### For wandb on-premises or dedicated cloud users
Artifact registration is required.
Below, the process of registering data to wandb's Artifacts is described for reference.
```bash
python3 scripts/upload_mtbench_question.py -e <wandb/entity> -p <wandb/project> -f "your question path"
python3 scripts/upload_mtbench_prompt.py -e <wandb/entity> -p <wandb/project> -f "your prompt path"
python3 scripts/upload_mtbench_referenceanswer.py -e <wandb/entity> -p <wandb/project> -f "your reference answer path"
```

## LCTG
Please adhere to the LCTG terms of use regarding data utilization.
#### For wandb multinant SaaS users
The artifact has been registered and the paths have been already put in the default config files.
The artifact path is in `configs/base-config.yaml`

### For wandb on-premises or dedicated cloud users
Artifact registration is required.
1. Please [download](https://docs.wandb.ai/guides/artifacts/download-and-use-an-artifact) the dataset from the artifact path in `configs/base-config.yaml` by logining on wandb multinant SaaS. When the official LCTG bench repository is released, you can download the file from there too.
2. Upload the dataset with 
```bash
python3 scripts/data_uploader.upload_dataset.py -e <your wandb entity> -p <your wandb project> -d <pass of LCTG dataset> -n lctg
```

## JBBQ
Please adhere to the JBBQ terms of use regarding data utilization.

**Manual upload required for all users, because the dataset is prohibited to distribute.**

1. The dataset can be downloaded from [JBBQ github repository](https://github.com/ynklab/JBBQ_data?tab=readme-ov-file). 
2. Upload the dataset with
```bash
python3 scripts/uploader/upload_jbbq.py -e <wandb/entity> -p <wandb/project>  -d <jbbq dataset path> -n jbbq
```

## LINE Yahoo Inappropriate Speech Evaluation Dataset
Please adhere to the LINE Yahoo Inappropriate Speech Evaluation Dataset" terms of use regarding data utilization.
