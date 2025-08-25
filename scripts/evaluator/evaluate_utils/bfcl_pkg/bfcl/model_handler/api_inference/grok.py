import os

from .openai_completion import OpenAICompletionsHandler
from openai import OpenAI
from overrides import override


class GrokHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.client = OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=os.getenv("XAI_API_KEY"),
        )
        self.is_fc_model = "FC" in self.model_name

    def _add_reasoning_content_if_available(self, api_response: any, response_data: dict) -> None:
        """
        Grok models support reasoning content in the API response.
        This method delegates to the appropriate method based on whether the model is FC or prompting.
        """
        if self.is_fc_model:
            self._add_reasoning_content_if_available_FC(api_response, response_data)
        else:
            self._add_reasoning_content_if_available_prompting(api_response, response_data)

    @override
    def _parse_query_response_prompting(self, api_response: any) -> dict:
        response_data = super()._parse_query_response_prompting(api_response)
        self._add_reasoning_content_if_available(api_response, response_data)
        return response_data

    @override
    def _parse_query_response_FC(self, api_response: any) -> dict:
        response_data = super()._parse_query_response_FC(api_response)
        self._add_reasoning_content_if_available(api_response, response_data)
        return response_data
