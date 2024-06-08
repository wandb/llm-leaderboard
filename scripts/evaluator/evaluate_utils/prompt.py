import json
from jinja2 import Template
from config_singleton import WandbConfigSingleton

def get_system_message(system_message_intro: str, instruction: str):
    system_message = ""
    system_message += system_message_intro
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
        tokenizer_config = cfg.tokenizer_config
        if cfg.model.chat_template.startswith("mistralai/"):
            tokenizer_config.update({"raise_exception": lambda _: ""})
        chat_template = Template(cfg.tokenizer_config.chat_template)
        conversation_prompt = chat_template.render(
            messages=messages, add_generation_prompt=True, **tokenizer_config
        )
    else:
        conversation_prompt = json.dumps(
            messages,
            ensure_ascii=False,
            indent=2,
        )
    return conversation_prompt
