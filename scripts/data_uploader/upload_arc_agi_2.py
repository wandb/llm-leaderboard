import wandb
from argparse import ArgumentParser
import os

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
    dataset_artifact = wandb.Artifact(name="arc_agi_2_public_eval",
                                    type="dataset",
                                    metadata={"version":args.dataset_version},
                                    description="This dataset is based on version {}".format(args.dataset_version))

    dataset_artifact.add_file(os.path.join(args.file_path, "evaluation.txt"))
    dataset_artifact.add_dir(os.path.join(args.file_path, "evaluation"), name="evaluation")
    run.log_artifact(dataset_artifact)
