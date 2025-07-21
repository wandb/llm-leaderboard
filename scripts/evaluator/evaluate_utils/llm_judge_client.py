import os
from llm_inference_adapter import AzureOpenAIResponsesClient, OpenAIResponsesClient


def get_openai_judge_client(model: str, **kwargs):
    """Get OpenAI client based on environment configuration"""
    api_type = os.environ.get('OPENAI_API_TYPE', 'openai')
    
    if api_type == "azure":
        if model.startswith("azure-"):
            model = model[6:]
        return AzureOpenAIResponsesClient(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2023-07-01-preview",
            azure_deployment=model,
            **kwargs,
        )
    else:
        return OpenAIResponsesClient(
            model=model,
            **kwargs,
        )
