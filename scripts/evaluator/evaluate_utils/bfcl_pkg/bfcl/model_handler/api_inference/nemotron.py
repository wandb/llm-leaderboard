import re
import os
from ..model_style import ModelStyle
from .nvidia import NvidiaHandler
from ..utils import (
    combine_consecutive_user_prompts,
    convert_system_prompt_into_user_prompt,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    func_doc_language_specific_pre_processing,
)
from openai import OpenAI
from overrides import override


class NemotronHandler(NvidiaHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
        )

    def _format_system_prompt(self, prompts, functions, test_category):
        """Format the system prompt for Nemotron with function definitions."""
        formatted_prompt = []
        
        for prompt in prompts:
            if prompt.get("role") == "system":
                # Build function definitions string
                func_definitions = []
                for func in functions:
                    func_def = f"def {func['name']}("
                    params = []
                    if 'parameters' in func and 'properties' in func['parameters']:
                        for param_name, param_info in func['parameters']['properties'].items():
                            param_type = param_info.get('type', 'Any')
                            params.append(f"{param_name}: {param_type}")
                    func_def += ", ".join(params) + "):\n"
                    func_def += f'    """{func.get("description", "")}\n'
                    
                    # Add parameter descriptions
                    if 'parameters' in func and 'properties' in func['parameters']:
                        for param_name, param_info in func['parameters']['properties'].items():
                            param_desc = param_info.get('description', '')
                            if param_desc:
                                func_def += f"    {param_name}: {param_desc}\n"
                    func_def += '    """\n    pass\n'
                    func_definitions.append(func_def)
                
                functions_str = "\n\n".join(func_definitions)
                
                system_content = prompt["content"] + f"\n\nYou have access to the following functions:\n\n{functions_str}\n\nWhen you need to call a function, format your response as:\n<TOOLCALL>\nfunction_name(param1=value1, param2=value2)\n</TOOLCALL>"
                
                formatted_prompt.append({
                    "role": "system",
                    "content": system_content
                })
            else:
                formatted_prompt.append(prompt)
        
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """Pre-process the input query and format it for the Nemotron model."""
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        # Pre-process functions based on language
        functions = func_doc_language_specific_pre_processing(functions, test_category)

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = convert_system_prompt_into_user_prompt(
                test_entry["question"][round_idx]
            )
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        test_entry["question"][0] = self._format_system_prompt(
            test_entry["question"][0], functions, test_category
        )

        # Return empty message list - messages will be added incrementally
        return {"message": []}

    @override
    def decode_ast(self, result, language="Python"):
        """Extract function calls from the Nemotron XML format."""
        # Extract content between TOOLCALL tags
        toolcall_match = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", result, re.DOTALL)
        if not toolcall_match:
            return []

        # Get the function call string
        func_call_str = toolcall_match.group(1)

        return default_decode_ast_prompting(func_call_str, language)

    @override
    def decode_execute(self, result, language="Python"):
        """Convert Nemotron response to executable function calls."""
        # Extract content between TOOLCALL tags
        toolcall_match = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", result, re.DOTALL)
        if not toolcall_match:
            return []

        # Get the function call string
        func_call_str = toolcall_match.group(1)

        return default_decode_execute_prompting(func_call_str, language) 