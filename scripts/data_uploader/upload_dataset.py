import wandb
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
    "-n",
    "--dataset_name",
    type=str,
    required=True
)
# choose from jaster, lctg

parser.add_argument(
    "-d",
    "--dataset_folder",
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
    dataset_artifact = wandb.Artifact(name=args.dataset_name,
                                    type="dataset", 
                                    metadata={"version":args.dataset_version})
    dataset_artifact.add_dir(args.dataset_folder,name=args.dataset_name)
    run.log_artifact(dataset_artifact)