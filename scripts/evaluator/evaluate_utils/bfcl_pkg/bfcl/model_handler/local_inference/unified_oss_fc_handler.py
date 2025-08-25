import json
import time
import traceback
from typing import Any
import warnings

from .base_oss_handler import OSSHandler
from ..utils import (
    convert_to_tool,
    convert_to_function_call,
    func_doc_language_specific_pre_processing,
)
from ...constants.type_mappings import GORILLA_TO_OPENAPI
from ..model_style import ModelStyle
from overrides import override
from config_singleton import WandbConfigSingleton

class UnifiedOSSFCHandler(OSSHandler):
    """
    統合OSSハンドラー（FC版）

    Tokenizerがtools引数に対応しており、vLLMのtool_call_parserに対応しているモデル用のOpenAI互換共通ハンドラー
    https://docs.vllm.ai/en/stable/features/tool_calling.html
    """

    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.is_fc_model = True
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config

        if cfg.api in ["vllm", "vllm-docker"]:
            assert cfg.vllm.tool_call_parser is not None, "cfg.vllm.tool_call_parser is not set"
            assert cfg.vllm.enable_auto_tool_choice, "cfg.vllm.enable_auto_tool_choice is false"
        elif cfg.api == "openai-compatible":
            warnings.warn(
                "UnifiedOSSFCHandler only supports vLLM server but cfg.api is set to openai-compatible. "
                "Make sure to cfg.base_url is set to vLLM server launched with tool_call_parser/enable_auto_tool_choice. "
                "See: https://docs.vllm.ai/en/stable/features/tool_calling.html"
            )

        print(f"[UnifiedOSSFCHandler] 初期化完了 - モデル: {model_name}")

    @override
    async def inference_async(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool):
        # FC model
        if "multi_turn" in test_entry["id"]:
            return await self.inference_multi_turn_FC_async(
                test_entry, include_input_log, exclude_state_log
            )
        else:
            return await self.inference_single_turn_FC_async(test_entry, include_input_log)
