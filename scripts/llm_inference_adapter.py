import os
from config_singleton import WandbConfigSingleton
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import VLLMOpenAI

def get_llm_inference_engine():
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    api_type = cfg.api

    if api_type == "vllm":
        # vLLMサーバーを起動
        from vllm_server import start_vllm_server
        start_vllm_server()

        # LangChainのVLLMインテグレーションを使用
        llm = ChatOpenAI(
            model=cfg.model.pretrained_model_name_or_path,
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1",
            **cfg.generator,
        )
        return llm
        # llm = VLLMOpenAI(
        #     openai_api_key="EMPTY",
        #     openai_api_base="http://localhost:8000/v1",
        #     model_name=cfg.model.pretrained_model_name_or_path,
        #     **cfg.generator,
        # )
        # return llm

    elif api_type == "openai":
        # LangChainのOpenAIインテグレーションを使用
        llm = ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )
        return llm

    elif api_type == "azure-openai":
        # Azure OpenAI APIの設定
        # ここにAnthropic APIの初期化コードを追加
        pass

    elif api_type == "anthropic":
        # Anthropic APIの設定
        # ここにAnthropic APIの初期化コードを追加
        pass

    elif api_type == "google":
        # Google APIの設定
        # ここにGoogle APIの初期化コードを追加
        pass

    elif api_type == "cohere":
        # Cohere APIの設定
        # ここにCohere APIの初期化コードを追加
        pass

    elif api_type == "bedrock":
        # AWS APIの設定
        # ここにAmazon Bedrock APIの初期化コードを追加
        pass

    else:
        raise ValueError(f"Unsupported API type: {api_type}")