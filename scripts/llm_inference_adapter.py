import os
from config_singleton import WandbConfigSingleton
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_aws import ChatBedrock
from langchain_anthropic import ChatAnthropic

# from langchain_cohere import Cohere

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
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1",
            model_name=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "openai":
        # LangChainのOpenAIインテグレーションを使用
        llm = ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "mistral":
        # LangChainのMistralAIインテグレーションを使用
        llm = ChatMistralAI(
            model=cfg.model.pretrained_model_name_or_path, 
            anthropic_api_key=os.environ["MISTRAL_API_KEY"],
            **cfg.generator,
        )

    elif api_type == "google":
        # LangChainのGoogleGenerativeAIインテグレーションを使用
        llm = ChatGoogleGenerativeAI(
            model=cfg.model.pretrained_model_name_or_path,
            api_key=os.environ["GOOGLE_API_KEY"],
            **cfg.generator,
            max_output_tokens = cfg.generator.get("max_tokens"),
        )
        # safety_settings_NONE = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        # ]
        # llm.client = genai.GenerativeModel(
        #     model_name=cfg.model.pretrained_model_name_or_path, 
        #     safety_settings=safety_settings_NONE
        # )

    elif api_type == "bedrock":
        # LangChainのBedrockインテグレーションを使用
        llm = ChatBedrock(
            region_name=os.environ["AWS_DEFAULT_REGION"],
            model_id=cfg.model.pretrained_model_name_or_path,
            model_kwargs=cfg.generator,
        )

    elif api_type == "anthropic":
        # LangChainのAnthropicインテグレーションを使用
        llm = ChatAnthropic(
            model=cfg.model.pretrained_model_name_or_path, 
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            **cfg.generator,
        )

    # elif api_type == "azure-openai":
    #     llm = AzureChatOpenAI(
    #         api_key=os.environ["OPENAI_API_KEY"],
    #         # api_base=os.environ["OPENAI_API_BASE"],
    #         # api_version=os.environ["OPENAI_API_VERSION"]
    #         api_version="2024-05-01-preview",
    #         model=cfg.model.pretrained_model_name_or_path,
    #         **cfg.generator,
    #     )

    # elif api_type == "cohere":
    #     llm = Cohere(
    #         model=cfg.model.pretrained_model_name_or_path,
    #         cohere_api_key=os.environ["COHERE_API_KEY"],
    #         **cfg.generator,
    #     )

    else:
        raise ValueError(f"Unsupported API type: {api_type}")
    
    return llm