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


parser.add_argument(
    "-f",
    "--file_path",
    type=str,
    required=True
)
args = parser.parse_args()

with wandb.init(entity=args.entity, project=args.project, job_type="upload_data") as run:
    dataset_artifact = wandb.Artifact(name="mtbench_ja_referenceanswer", 
                                    type="dataset", 
                                    metadata={"version":args.dataset_version},
                                    description="This dataset is based on version {}".format(args.dataset_version))
    
    # track lineage
    run.use_artifact('wandb-japan/llm-leaderboard/mtbench_ja_prompt:v0', type='dataset')
    run.use_artifact('wandb-japan/llm-leaderboard/mtbench_ja_question:v0', type='dataset')

    dataset_artifact.add_file(args.file_path)

    run.log_artifact(dataset_artifact)