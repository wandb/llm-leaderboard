from config_singleton import WandbConfigSingleton
import json, os
from pathlib import Path
from huggingface_hub import HfApi


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