import time

from config_singleton import WandbConfigSingleton


API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

def chat_completion_vllm(model, conv, temperature, max_tokens):
    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://0.0.0.0:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=conv,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            output = response.choices[0].message.content
            break
        except Exception as e:  # 修正: openai.error.OpenAIError から Exception へ
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def evaluate():
    config = WandbConfigSingleton.get_instance().config

    model = config.model.pretrained_model_name_or_path
    conv = [
        {"role": "user", "content": "Who is Elon Musk?"},
    ]
    temperature = config.generator.temperature
    max_tokens = 64
    output = chat_completion_vllm(
        model=model,
        conv=conv,
        temperature=temperature,
        max_tokens=max_tokens
    )
    print(output)
    return