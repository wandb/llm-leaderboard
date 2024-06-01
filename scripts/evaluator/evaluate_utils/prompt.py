import json
from pathlib import Path

from jinja2 import Template

from config_singleton import WandbConfigSingleton


def get_system_message_intro(language: str) -> str:
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config

    custom_system_message = cfg.get(f"custom_system_message_{language}", None)
    if custom_system_message is not None:
        return custom_system_message
    elif language == "ja":
        return "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    elif language == "en":
        return "Here is a combination of instructions explaining the task and inputs with context. Please write a response that adequately meets the request."
    else:
        raise ValueError(f"Invalid language: {language}")


def get_system_message(language: str, instruction: str):
    system_message = ""
    system_message += get_system_message_intro(language=language)
    system_message += "\n"
    system_message += instruction
    return system_message


def get_few_shot_messages(target_dataset_path: str, num_few_shots: int):
    dataset_json_name = target_dataset_path.name
    target_few_shot_path = (
        target_dataset_path.resolve().parent.parent / "train" / dataset_json_name
    )
    assert (
        target_few_shot_path.exists()
    ), f"Wrong path {target_few_shot_path.resolve()}, can not extract few-shot samples"
    samples = json.loads(target_few_shot_path.read_text(encoding="utf-8"))["samples"]

    few_shot_messages = []
    for i in range(num_few_shots):
        few_shot_messages.append({"role": "user", "content": samples[i]["input"]})
        few_shot_messages.append({"role": "assistant", "content": samples[i]["output"]})
    return few_shot_messages


def apply_chat_template(messages: list[dict[str, str]]) -> str:
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config

    if cfg.api == "vllm":
        chat_template_path = Path(f"chat_templates/{cfg.model.chat_template}.jinja")
        if not chat_template_path.exists():
            raise ValueError(f"Chat template {chat_template_path} not found")
        else:
            with chat_template_path.open(encoding="utf-8") as f:
                chat_template = Template(f.read())
        tokenizer_config = cfg.tokenizer_config
        conversation_prompt = chat_template.render(
            messages=messages, add_generation_prompt=True, **tokenizer_config
        )
    else:
        chat_template_path = Path(f"chat_templates/defaults/template_alpaca_en.jinja")
        with chat_template_path.open(encoding="utf-8") as f:
            chat_template = Template(f.read())
        conversation_prompt = chat_template.render(
            messages=messages, add_generation_prompt=True
        )
    return conversation_prompt
