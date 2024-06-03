import wandb
import sys

dest = sys.argv[1] # user/project/artifact name
data_path = sys.argv[2]

entity, project, name = dest.split("/")

with wandb.init(entity=entity, project=project, job_type="upload_data") as run:
    dataset_artifact = wandb.Artifact(name=name, type="dataset")
    dataset_artifact.add_dir(data_path,name=name)
    run.log_artifact(dataset_artifact)
