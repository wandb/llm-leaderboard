import gc
import json
import torch

import wandb
import pandas as pd

from huggingface_hub import HfApi
from pathlib import Path
import os


def cleanup_gpu():
    """
    Function to clean up GPU memory
    """
    # Remove references to all CUDA objects
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            del obj
    gc.collect()
    torch.cuda.empty_cache()


def read_wandb_table(
    table_name: str,
    run: object = None,
    entity: str = None,
    project: str = None,
    run_id: str = None,
    version: str = "latest",
) -> pd.DataFrame:
    if run is None:
        assert isinstance(entity, str), "entity is not string"
        assert isinstance(project, str), "project is not string"
        assert isinstance(run_id, str), "run_id is not string"
    else:
        entity = run.entity
        project = run.project
        run_id = run.id
    artifact_path = f"{entity}/{project}/run-{run_id}-{table_name}:{version}"
    artifact = run.use_artifact(artifact_path)
    artifact_dir = artifact.download()
    with open(f"{artifact_dir}/{table_name}.table.json") as f:
        tjs = json.load(f)
    output_table = wandb.Table.from_json(json_obj=tjs, source_artifact=artifact)
    output_df = pd.DataFrame(data=output_table.data, columns=output_table.columns)
    return output_df

def download_tokenizer_config(repo_id: str):
    api = HfApi()
    file_path = api.hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer_config.json",
        revision="main",
        use_auth_token=os.getenv("HUGGING_TOKEN"),
    )

    with Path(file_path).open("r") as f:
        tokenizer_config = json.load(f)
    
    return tokenizer_config