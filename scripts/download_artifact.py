import wandb
import sys

artifact = sys.argv[1]
project = sys.argv[2]

run = wandb.init(project=project, job_type="download")

artifact = run.use_artifact(artifact)
datadir = artifact.download()
print(datadir)
# that's it

