from overrides import EnforceOverrides

from ..openai_compatible_handler import OpenAICompatibleHandler


class WandBInferenceHandler(OpenAICompatibleHandler, EnforceOverrides):
    """OpenAI-compatible handler for W&B Inference.

    The actual model id, base URL, project, and credentials are resolved from the
    main leaderboard YAML through llm_inference_adapter.OpenAIClient.
    """

    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
