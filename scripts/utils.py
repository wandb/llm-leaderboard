import os
import gc
import json
import time
from typing import Any

from huggingface_hub import HfApi
import pandas as pd
from pathlib import Path
import torch
import wandb
from tenacity import retry, stop_after_attempt, wait_fixed
import questionary

from config_singleton import WandbConfigSingleton


@retry(stop=stop_after_attempt(8), wait=wait_fixed(5))
def read_wandb_table_light(
    table_name: str,
    run: object,
    run_path: str = None,
    entity: str = None,
    project: str = None,
    run_id: str = None,
    version: str = "latest",
) -> pd.DataFrame:
    """
    軽量版: アーティファクト全体をダウンロードせず、必要な table.json のみ取得してDataFrame化する。

    - Mediaや画像ファイルの実体は取得しない（参照のみ）。
    - 大規模アーティファクトでもネットワーク負荷を最小化。
    """
    if run_path is not None:
        entity, project, run_id = run_path.split("/")
    elif entity is None or project is None or run_id is None:
        entity = run.entity
        project = run.project
        run_id = run.id

    artifact_path = f"{entity}/{project}/run-{run_id}-{table_name}:{version}"
    artifact = run.use_artifact(artifact_path)

    # 必要なJSONファイルだけを取得
    table_json_path_ref = artifact.get_entry(f"{table_name}.table.json")
    # 直接ファイルパスを返さない実装もあるため、ローカルに保存
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        local_json_path = table_json_path_ref.download(root=tmpdir)
        with open(local_json_path, "r") as f:
            tjs = json.load(f)

    # wandb.Tableを経由せず、直接DataFrame化してメディア解決を避ける
    output_df = pd.DataFrame(data=tjs.get("data", []), columns=tjs.get("columns", []))
    return output_df


@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def read_wandb_table_throttled(
    table_name: str,
    run: object,
    run_path: str = None,
    entity: str = None,
    project: str = None,
    run_id: str = None,
    version: str = "latest",
    sleep_sec: float = 0.05,
) -> pd.DataFrame:
    """
    すべてのアーティファクトファイルを順次ダウンロード（スロットル付き）した上で、テーブルを読み込む。
    - ダウンロード内容は従来どおりフル（画像等を含む）だが、一括ではなく逐次取得して負荷を低減。
    - blend run 専用での利用を想定。
    """
    if run_path is not None:
        entity, project, run_id = run_path.split("/")
    elif entity is None or project is None or run_id is None:
        entity = run.entity
        project = run.project
        run_id = run.id

    artifact_path = f"{entity}/{project}/run-{run_id}-{table_name}:{version}"
    artifact = run.use_artifact(artifact_path)

    # マニフェストに含まれる全ファイルを順次ダウンロード
    entries = getattr(artifact, "manifest").entries
    total_files = len(entries)
    print(f"[blend] Starting sequential download: {artifact_path} ({total_files} files)")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        for idx, rel_path in enumerate(entries.keys(), start=1):
            artifact.get_entry(rel_path).download(root=tmpdir)
            if idx % 50 == 0 or idx == total_files:
                print(f"[blend] Downloaded {idx}/{total_files}: {rel_path}")
            if sleep_sec and sleep_sec > 0:
                time.sleep(sleep_sec)

        # テーブルJSONを読み込み
        table_json_path = Path(tmpdir) / f"{table_name}.table.json"
        with table_json_path.open("r") as f:
            tjs = json.load(f)

    # DataFrame化（Media実体はローカルに順次ダウン済み）
    output_df = pd.DataFrame(data=tjs.get("data", []), columns=tjs.get("columns", []))
    print(f"[blend] Completed: {artifact_path}")
    return output_df

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

def wait_for_gpu_ready(timeout: int = 30):
    """
    GPUが利用可能になるまで待機
    """
    if not torch.cuda.is_available():
        return

    import time
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # 少量のメモリを確保してGPUが使えることを確認
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.synchronize()
            return
        except RuntimeError as e:
            if "out of memory" in str(e):
                cleanup_gpu()
                time.sleep(1)
            else:
                raise

    raise TimeoutError(f"GPU did not become ready within {timeout} seconds")

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
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    model_local_path = cfg.model.get("local_path", None)

    if model_id is None and chat_template_name is None:
        model_id = cfg.model.pretrained_model_name_or_path
        chat_template_name = cfg.model.get("chat_template")

    # get tokenizer_config
    if model_local_path is not None:
        with (model_local_path / "tokenizer_config.json").open() as f:
            tokenizer_config = json.load(f)
    else:
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

    # add chat_template to tokenizer_config
    tokenizer_config.update({"chat_template": chat_template})

    return tokenizer_config

def paginate_choices(choices, page_size=36):
    page = 0
    while True:
        start = page * page_size
        end = start + page_size
        current_choices = choices[start:end]
        
        if page > 0:
            current_choices.append("< Previous Page")
        
        if end < len(choices):
            current_choices.append("Next Page >")
            
        selected = questionary.select(
            f"Select config (Page {page + 1})",
            choices=current_choices
        ).ask()
        
        if selected == "Next Page >":
            page += 1
        elif selected == "< Previous Page":
            page -= 1
        else:
            return selected