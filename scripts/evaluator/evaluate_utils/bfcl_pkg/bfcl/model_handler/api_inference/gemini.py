import os
import time
import logging
import warnings

from ...constants.type_mappings import GORILLA_TO_OPENAPI
from ..base_handler import BaseHandler
from ..model_style import ModelStyle
from ..utils import (
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    extract_system_prompt,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    GenerateContentConfig,
    Part,
    ThinkingConfig,
    Tool,
    HttpOptions,
)
from config_singleton import WandbConfigSingleton



class GeminiHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.GOOGLE
        instance = WandbConfigSingleton.get_instance()
        self.cfg = instance.config
        self.api_model_name = self.cfg.model.pretrained_model_name_or_path
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable must be set for Gemini models"
            )
        # Use v1beta API for Gemini 2.5 models (function calling support)
        http_options = HttpOptions(api_version="v1beta")
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        
        # Completely disable Google GenAI logging
        for logger_name in ['google.genai.types', 'google.genai.models', 'google.genai', 'google']:
            logger = logging.getLogger(logger_name)
            logger.disabled = True
            logger.propagate = False
            # Remove all handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        # Set environment variables to suppress warnings
        os.environ['GOOGLE_GENAI_SUPPRESS_WARNINGS'] = 'true'
        os.environ['GOOGLE_GENAI_QUIET'] = 'true'
        os.environ['GOOGLE_GENAI_SILENT'] = 'true'
        
        # Suppress all warnings from Google libraries
        warnings.filterwarnings("ignore", category=UserWarning, module="google")
        warnings.filterwarnings("ignore", message=".*AFC.*")
        warnings.filterwarnings("ignore", message=".*function_call.*")
        warnings.filterwarnings("ignore", message=".*thought_signature.*")

        # please set thinking_budget (more than 0 or -1) in config file if you use gemini-2.5-pro
        self.thinking_budget = getattr(self.cfg.model, 'thinking_budget', 0)
        

    @staticmethod
    def _substitute_prompt_role(prompts: list[dict]) -> list[dict]:
        # Allowed roles: user, model
        for prompt in prompts:
            if prompt["role"] == "user":
                prompt["role"] = "user"
            elif prompt["role"] == "assistant":
                prompt["role"] = "model"

        return prompts

    def decode_ast(self, result, language="Python"):
        if "FC" not in self.model_name:
            result = result.replace("```tool_code\n", "").replace("\n```", "")
            return default_decode_ast_prompting(result, language)
        else:
            if type(result) is not list:
                result = [result]
            return result

    def decode_execute(self, result):
        if "FC" not in self.model_name:
            result = result.replace("```tool_code\n", "").replace("\n```", "")
            return default_decode_execute_prompting(result)
        else:
            func_call_list = []
            for function_call in result:
                for func_name, func_args in function_call.items():
                    func_call_list.append(
                        f"{func_name}({','.join([f'{k}={repr(v)}' for k, v in func_args.items()])})"
                    )
            return func_call_list

    # We can't retry on ClientError because it's too broad.
    # Both rate limit and invalid function description will trigger google.genai.errors.ClientError
    @retry_with_backoff(error_message_pattern=r".*RESOURCE_EXHAUSTED.*")
    def generate_with_backoff(self, **kwargs):
        """Generate content with retry logic and enhanced error handling."""
        start_time = time.time()
        
        try:
            # Log the request parameters for debugging
            model = kwargs.get('model', 'unknown')
            contents_count = len(kwargs.get('contents', []))
            has_tools = 'config' in kwargs and hasattr(kwargs['config'], 'tools') and kwargs['config'].tools
            has_thinking = 'config' in kwargs and hasattr(kwargs['config'], 'thinking_config')
            
            # Only log essential info
            if not has_tools:
                print(f"Warning: No tools provided for function calling")
            
            # Minimal debug info (only show if there's an issue)
            if 'config' in kwargs:
                config = kwargs['config']
                if hasattr(config, 'tools') and not config.tools:
                    print(f"Warning: Tools field is empty")
            
            api_response = self.client.models.generate_content(**kwargs)
            
            # Only log response info if there's an issue
            if hasattr(api_response, 'candidates') and api_response.candidates:
                candidate = api_response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    print(f"Warning: Empty response content")
            
            end_time = time.time()
            return api_response, end_time - start_time
            
        except Exception as e:
            print(f"Error in generate_with_backoff: {type(e).__name__}: {e}")
            # Re-raise the exception for retry logic
            raise

    def _create_generate_config(self, system_prompt=None, tools=None):
        """Create a common GenerateContentConfig with thinking_budget and other settings."""
        # Create base config (AFC will be handled by the API defaults)
        config = GenerateContentConfig()
        
        # Add thinking configuration
        if self.thinking_budget > 0 or self.thinking_budget == -1:
            config.thinking_config = ThinkingConfig(
                include_thoughts=True,
                thinking_budget=self.thinking_budget
            )
        
        # Add system instruction if provided
        if system_prompt:
            config.system_instruction = system_prompt
            
        # Add tools if provided
        if tools:
            # Create Tool objects with function declarations
            # According to Gemini API docs, tools should be a list of Tool objects
            tool_objects = []
            for tool in tools:
                if isinstance(tool, dict):
                    # Convert dict format to Tool object
                    tool_objects.append(Tool(function_declarations=[tool]))
                else:
                    # Assume it's already a Tool object
                    tool_objects.append(tool)
            
            # Set tools directly as per official documentation
            try:
                config.tools = tool_objects
            except Exception as e:
                # Try alternative approach if tools field fails
                try:
                    if hasattr(config, 'tool_config'):
                        config.tool_config = tool_objects
                except Exception:
                    pass  # Silently fail if both approaches don't work
            
        # Add safety settings from config if available
        if hasattr(self.cfg.model, 'safety_settings') and self.cfg.model.safety_settings:
            try:
                # Convert dict format to SafetySetting objects if needed
                safety_settings = []
                for setting in self.cfg.model.safety_settings:
                    if isinstance(setting, dict):
                        safety_settings.append(
                            types.SafetySetting(
                                category=setting['category'],
                                threshold=setting['threshold']
                            )
                        )
                    else:
                        safety_settings.append(setting)
                config.safety_settings = safety_settings
            except Exception as e:
                print(f"Warning: Failed to configure safety settings: {e}")
        
        # Add temperature directly to config if supported
        # Note: Some versions of the SDK may not support direct temperature setting
        if hasattr(self, 'temperature') and self.temperature is not None:
            try:
                # Try to set temperature directly on the config
                if hasattr(config, 'temperature'):
                    config.temperature = self.temperature
                else:
                    print(f"Warning: Temperature {self.temperature} cannot be set directly on config")
            except Exception as e:
                print(f"Warning: Failed to set temperature: {e}")
            
        # Only log if there's an issue
        if not hasattr(config, 'temperature'):
            print("Warning: Temperature field not available on config")
            
        return config

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        inference_data["inference_input_log"] = {
            "message": repr(inference_data["message"]),
            "tools": inference_data["tools"],
            "system_prompt": inference_data.get("system_prompt", None),
        }

        config = self._create_generate_config(
            system_prompt=inference_data.get("system_prompt"),
            tools=inference_data["tools"] if len(inference_data["tools"]) > 0 else None
        )

        return self.generate_with_backoff(
            model=self.model_name.replace("-FC", ""),
            contents=inference_data["message"],
            config=config,
        )

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = GeminiHandler._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        inference_data["message"] = []

        system_prompt = extract_system_prompt(test_entry["question"][0])
        if system_prompt:
            inference_data["system_prompt"] = system_prompt
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: any) -> dict:
        tool_call_func_names = []
        fc_parts = []
        text_parts = []
        reasoning_content = []

        if (
            len(api_response.candidates) > 0
            and api_response.candidates[0].content
            and api_response.candidates[0].content.parts
            and len(api_response.candidates[0].content.parts) > 0
        ):
            response_function_call_content = api_response.candidates[0].content

            for part in api_response.candidates[0].content.parts:
                # Check for function calls according to Gemini API docs
                if hasattr(part, 'function_call') and part.function_call:
                    # Verify the function call has a valid name
                    if part.function_call.name and part.function_call.name.strip():
                        part_func_name = part.function_call.name
                        part_func_args = part.function_call.args
                        
                        # Convert args to dict format
                        if hasattr(part_func_args, 'items'):
                            part_func_args_dict = {k: v for k, v in part_func_args.items()}
                        else:
                            # Handle case where args might be a string or other format
                            part_func_args_dict = {"args": part_func_args}
                        
                        fc_parts.append({part_func_name: part_func_args_dict})
                        tool_call_func_names.append(part_func_name)
                        # Function call detected successfully
                        pass
                    else:
                        print("Warning: Function call detected but name is empty")
                        
                # Check for reasoning content (thoughts)
                elif hasattr(part, 'thought') and part.thought:
                    reasoning_content.append(part.text)
                    # Reasoning content detected
                    pass
                    
                # Regular text content
                elif hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
                    # Text content detected
                    pass
                    
                # Log unexpected part types for debugging
                else:
                    # Unexpected part type - only log if debugging is needed
                    pass

        else:
            response_function_call_content = Content(
                role="model",
                parts=[
                    Part(text="The model did not return any response."),
                ],
            )

        model_responses = fc_parts if fc_parts else text_parts

        # Handle usage_metadata for Gemini 2.5 models
        input_token = 0
        output_token = 0
        thoughts_token = 0
        total_token = 0
        
        if hasattr(api_response, 'usage_metadata') and api_response.usage_metadata is not None:
            input_token = getattr(api_response.usage_metadata, 'prompt_token_count', 0) or 0
            output_token = getattr(api_response.usage_metadata, 'candidates_token_count', 0) or 0
            thoughts_token = getattr(api_response.usage_metadata, 'thoughts_token_count', 0) or 0
            total_token = getattr(api_response.usage_metadata, 'total_token_count', 0) or 0
        
        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": response_function_call_content,
            "tool_call_func_names": tool_call_func_names,
            "reasoning_content": "\n".join(reasoning_content),
            "input_token": input_token,
            "output_token": output_token,
            "thoughts_token": thoughts_token,
            "total_token": total_token,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        for message in first_turn_message:
            inference_data["message"].append(
                Content(
                    role=message["role"],
                    parts=[
                        Part(text=message["content"]),
                    ],
                )
            )
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        return self.add_first_turn_message_FC(inference_data, user_message)

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """Add tool execution results to the conversation history.
        
        According to Gemini API docs, tool responses should be added as user messages
        with function_response parts containing the results.
        """
        if not execution_results or not model_response_data.get("tool_call_func_names"):
            print("Warning: No execution results or tool call names to add")
            return inference_data
            
        # Create tool response parts for each executed function
        tool_response_parts = []
        for execution_result, tool_call_func_name in zip(
            execution_results, model_response_data["tool_call_func_names"]
        ):
            try:
                # Create function response part according to Gemini API docs
                tool_response_part = Part.from_function_response(
                    name=tool_call_func_name,
                    response={
                        "result": execution_result,
                    },
                )
                tool_response_parts.append(tool_response_part)
                # Tool response added successfully
                pass
            except Exception as e:
                # Fallback: create a simple text part
                tool_response_parts.append(Part(text=f"Tool {tool_call_func_name} result: {execution_result}"))

        # Create content object with all tool responses
        if tool_response_parts:
            tool_response_content = Content(role="user", parts=tool_response_parts)
            inference_data["message"].append(tool_response_content)
            # Tool responses added to conversation history
            pass
        else:
            print("Warning: No valid tool response parts created")

        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {
            "message": repr(inference_data["message"]),
            "system_prompt": inference_data.get("system_prompt", None),
        }

        config = self._create_generate_config(
            system_prompt=inference_data.get("system_prompt")
        )

        api_response = self.generate_with_backoff(
            model=self.model_name.replace("-FC", ""),
            contents=inference_data["message"],
            config=config,
        )
        return api_response

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = GeminiHandler._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )
        # Gemini has system prompt in a specific field
        system_prompt = extract_system_prompt(test_entry["question"][0])

        if system_prompt:
            return {"message": [], "system_prompt": system_prompt}
        else:
            return {"message": []}

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        """Parse the response from Gemini API for prompting methods.
        
        According to Gemini API docs, responses can contain:
        - Text content in parts
        - Reasoning content in thought parts
        - Function calls in function_call parts
        """
        if (
            len(api_response.candidates) > 0
            and api_response.candidates[0].content
            and api_response.candidates[0].content.parts
            and len(api_response.candidates[0].content.parts) > 0
        ):
            model_responses = ""
            reasoning_content = ""
            
            for part in api_response.candidates[0].content.parts:
                # Check for reasoning content (thoughts)
                if hasattr(part, 'thought') and part.thought:
                    reasoning_content = part.text
                    # Reasoning content detected in prompting
                    pass
                    
                # Check for function calls (should not happen in prompting but handle gracefully)
                elif hasattr(part, 'function_call') and part.function_call:
                    print(f"Warning: Function call detected in prompting response: {part.function_call.name}")
                    # In prompting mode, we don't expect function calls, but log them
                    
                # Regular text content
                elif hasattr(part, 'text') and part.text:
                    model_responses = part.text
                    # Text content detected in prompting
                    pass
                    
                # Log unexpected part types for debugging
                else:
                    # Unexpected part type in prompting - only log if debugging is needed
                    pass

        else:
            model_responses = "The model did not return any response."

        # Handle usage_metadata for Gemini 2.5 models
        input_token = 0
        output_token = 0
        thoughts_token = 0
        total_token = 0
        
        if hasattr(api_response, 'usage_metadata') and api_response.usage_metadata is not None:
            input_token = getattr(api_response.usage_metadata, 'prompt_token_count', 0) or 0
            output_token = getattr(api_response.usage_metadata, 'candidates_token_count', 0) or 0
            thoughts_token = getattr(api_response.usage_metadata, 'thoughts_token_count', 0) or 0
            total_token = getattr(api_response.usage_metadata, 'total_token_count', 0) or 0
        
        return {
            "model_responses": model_responses,
            "reasoning_content": reasoning_content,
            "input_token": input_token,
            "output_token": output_token,
            "thoughts_token": thoughts_token,
            "total_token": total_token,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        for message in first_turn_message:
            inference_data["message"].append(
                Content(
                    role=message["role"],
                    parts=[
                        Part(text=message["content"]),
                    ],
                )
            )
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        return self.add_first_turn_message_prompting(inference_data, user_message)

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            Content(
                role="model",
                parts=[
                    Part(text=model_response_data["model_responses"]),
                ],
            )
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        tool_message = Content(
            role="user",
            parts=[
                Part(text=formatted_results_message),
            ],
        )
        inference_data["message"].append(tool_message)
        return inference_data
