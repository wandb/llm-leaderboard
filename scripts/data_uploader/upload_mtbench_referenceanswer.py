import wandb
import requests
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "-e",
    "--entity",
    type=str,
    required=True
)
parser.add_argument(
    "-p",
    "--project",
    type=str,
    required=True
)

parser.add_argument(
    "-f",
    "--file_path",
    type=str,
    required=True
)
args = parser.parse_args()

# 2025-06-30
# Use swallow evaluation dataset : 
metadata = {
    "original_dataset": "https://github.com/swallow-llm/swallow-evaluation/tree/main/fastchat/fastchat/llm_judge/data/japanese_mt_bench",
}
description = "This is based on the refrence answer dataset developed by the evaluation team of Swallow project. URL: https://github.com/swallow-llm/swallow-evaluation/tree/main"

with wandb.init(entity=args.entity, project=args.project, job_type="upload_data") as run:
    dataset_artifact = wandb.Artifact(name="mtbench_ja_referenceanswer", 
                                    type="dataset", 
                                    metadata=metadata,
                                    description=description)
    
    # track lineage
    run.use_artifact('wandb-japan/llm-leaderboard/mtbench_ja_prompt:v0', type='dataset')
    run.use_artifact('wandb-japan/llm-leaderboard/mtbench_ja_question:v0', type='dataset')

    dataset_artifact.add_file(args.file_path)

    run.log_artifact(dataset_artifact)