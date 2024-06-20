import logging
import json
from botocore.exceptions import ClientError
import boto3
from dataclasses import dataclass


from config_singleton import WandbConfigSingleton

@dataclass
class BedrockResponse:
    content: str

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_message(
    bedrock_runtime: boto3.client,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    generator_config: dict[str, float],
):
    # create body
    body_dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        **generator_config,
    }

    # handle system message
    if messages[0]["role"] == "system":
        body_dict.update({"messages": messages[1:], "system": messages[0]["content"]})
    else:
        body_dict.update({"messages": messages})

    # inference
    response = bedrock_runtime.invoke_model(
        body=json.dumps(body_dict), modelId=model_id
    )
    response_body = json.loads(response.get("body").read())

    return response_body


def chat_bedrock(messages: list[dict[str, str]], max_tokens: int):
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    try:
        bedrock_runtime = boto3.client(service_name="bedrock-runtime")
        model_id = cfg.model.pretrained_model_name_or_path
        ignore_keys = ["max_tokens"]
        generator_config = {k: v for k, v in cfg.generator.items() if not k in ignore_keys}

        response = generate_message(
            bedrock_runtime=bedrock_runtime,
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            generator_config=generator_config,
        )
        try:
            if response["content"]:
                return BedrockResponse(content=response["content"][0]["text"]), 0
            else:
                return BedrockResponse(content=""), 0

        except:
            print("--- prompt ---")
            print(messages)
            print("--- response ---")
            print(repr(response))
            raise ValueError

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
