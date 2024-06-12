import os
import gc
import json
from typing import Any

from huggingface_hub import HfApi
import pandas as pd
from pathlib import Path
import torch
import wandb
from tenacity import retry, stop_after_attempt, wait_fixed

from config_singleton import WandbConfigSingleton


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

@retry(stop=stop_after_attempt(10), wait=wait_fixed(10))
def read_wandb_table(
    table_name: str,
    run: object,
    run_path: str = None,
    entity: str = None,
    project: str = None,
    run_id: str = None,
    version: str = "latest",
) -> pd.DataFrame:
    if run_path is not None:
        entity, project, run_id = run_path.split("/")
    elif entity is None or project is None or run_id is None:
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


def hf_download(repo_id: str, filename: str):
    api = HfApi()
    file_path = api.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision="main",
        use_auth_token=os.getenv("HUGGING_TOKEN"),
    )

    with Path(file_path).open("r") as f:
        tokenizer_config = json.load(f)

    return tokenizer_config


def get_tokenizer_config(model_id=None, chat_template_name=None) -> dict[str, Any]:
    if model_id is None and chat_template_name is None:
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        model_id = cfg.model.pretrained_model_name_or_path
        chat_template_name = cfg.model.get("chat_template")
    assert isinstance(model_id, str) and isinstance(chat_template_name, str)

    # get tokenizer_config
    tokenizer_config = hf_download(
        repo_id=model_id,
        filename="tokenizer_config.json",
    )

    # chat_template from local
    local_chat_template_path = Path(f"chat_templates/{chat_template_name}.jinja")
    if local_chat_template_path.exists():
        with local_chat_template_path.open(encoding="utf-8") as f:
            chat_template = f.read()
    # chat_template from hf
    else:
        chat_template = hf_download(
            repo_id=chat_template_name, filename="tokenizer_config.json"
        ).get("chat_template")
        assert chat_template is not None

    # add chat_template to tokenizer_config
    tokenizer_config.update({"chat_template": chat_template})

    return tokenizer_config