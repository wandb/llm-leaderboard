import gc
import json
import torch

import wandb
import pandas as pd


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
    run: object,
    table_name: str,
    version: str = "latest",
    entity: str = None,
    project: str = None,
    run_id: str = None,
) -> pd.DataFrame:
    artifact_path = f"{entity}/{project}/run-{run.id}-{table_name}:{version}"
    artifact = run.use_artifact(artifact_path)
    artifact_dir = artifact.download()
    with open(f"{artifact_dir}/{table_name}.table.json") as f:
        tjs = json.load(f)
    output_table = wandb.Table.from_json(json_obj=tjs, source_artifact=artifact)
    output_df = pd.DataFrame(data=output_table.data, columns=output_table.columns)
    return output_df
