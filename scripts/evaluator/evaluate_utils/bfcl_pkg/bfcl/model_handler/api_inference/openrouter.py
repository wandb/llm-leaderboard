from overrides import EnforceOverrides
from ..openai_compatible_handler import OpenAICompatibleHandler

class OpenRouterHandler(OpenAICompatibleHandler, EnforceOverrides):
    def __init__(self, model_name, temperature) -> None:
        # temperatureは後方互換のため残しているがgenerator_configから取るので使用しない
        super().__init__(model_name, temperature)
        #self.model_name = model_name.replace("-OpenRouter", "")
