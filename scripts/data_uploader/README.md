# Dataset Preparation

## JASTER 
jaster is the Japanese evaluation dataset managed by llm-jp in llm-jp-eval.

- For wandb multinant SaaS users, the artifact has been registered and the paths have been already put in the default config files.
    - artifact path: `wandb-japan/llm-leaderboard3/jaster:v6`
- For wandb on-premises or dedicated cloud users, artifact registration is required.
    - Download the dataset from the above artifacts by logging on wandb multinant SaaS. or create jaster dataset by following the instruction in [llm-jp-eval/nejumi3-data](https://github.com/llm-jp/llm-jp-eval/tree/nejumi3-data).
    - Upload the dataset with 
        ```bash
        python3 scripts/data_uploader.upload_dataset.py -e <your wandb entity> -p <your wandb project> -d <pass of jaster dataset> -n jaster -v <version>
        ```

## MT-Bench
- For wandb multinant SaaS users, the artifact has been registered.

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
- For wandb on-premises or dedicated cloud users, artifact registration is required.
    
    Below, the process of registering data in wandb's Artifacts is described for reference 
    ```bash
    # register questions
    python3 scripts/upload_mtbench_question.py -e <wandb/entity> -p <wandb/project> -f "your path"


## LCTG
Please adhere to the LCTG terms of use regarding data utilization.
- For wandb multinant SaaS users, the artifact has been registered and the paths have been already put in the default config files.
    - artifact path: `wandb-japan/llm-leaderboard3/lctg:v0`
- For wandb on-premises or dedicated cloud users, artifact registration is required.
    - Please download the dataset from the above artifact pash by logging on wandb multinant SaaS.
    - Upload the dataset with 
    ```bash
    python3 scripts/data_uploader.upload_dataset.py -e <your wandb entity> -p <your wandb project> -d <pass of LCTG dataset> -n lctg
    ```

## JBBQ
Please adhere to the JBBQ terms of use regarding data utilization.
- Manual upload required.
- The dataset can be downloaded from [JBBQ github repository](https://github.com/ynklab/JBBQ_data?tab=readme-ov-file). 
- Upload the dataset with
    ```bash
    python3 scripts/uploader/upload_jbbq.py -e <wandb/entity> -p <wandb/project>  -d <jbbq dataset path> -n jbbq
    ```

## LINE Yahoo Inappropriate Speech Evaluation Dataset
Please adhere to the LINE Yahoo Inappropriate Speech Evaluation Dataset" terms of use regarding data utilization.
