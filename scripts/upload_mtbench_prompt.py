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
    "-v",
    "--dataset_version",
    type=str,
    required=True
)
args = parser.parse_args()

with wandb.init(entity=args.entity, project=args.project, job_type="upload_data") as run:
    dataset_artifact = wandb.Artifact(name="mtbench_prompt", 
                                    type="dataset", 
                                    metadata={"version":args.dataset_version},
                                    description="This dataset is based on version {}".format(args.dataset_version))


    url = "https://github.com/Stability-AI/FastChat/blob/jp-stable/fastchat/llm_judge/data/judge_ja_prompts.jsonl"
    response = requests.get(url)
    with open("judge_ja_prompts.jsonl", "wb") as file:
        file.write(response.content)
    dataset_artifact.add_file("judge_ja_prompts.jsonl")

    run.log_artifact(dataset_artifact)