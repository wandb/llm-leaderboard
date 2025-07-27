import importlib
import inspect
import json
import re
import copy
import sys
from pathlib import Path


# Get the directory containing func_source_code
FUNC_SOURCE_DIR = Path(__file__).parent / "func_source_code"

CLASS_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# These classes are stateless and do not require any initial configuration
STATELESS_CLASSES = [
    "MathAPI",
]

def get_multi_turn_test_case_id(test_entry_id: str) -> str:
    """
    Extract the test case ID from the test entry ID.
    Example: 'multi_turn_base_0' -> 'multi_turn_base'
    """
    return test_entry_id.rsplit("_", 1)[0]

def get_multi_turn_test_case_turn_index(test_entry_id: str) -> int:
    """
    Extract the turn index from the test entry ID.
    Example: 'multi_turn_base_0' -> 0
    """
    return int(test_entry_id.rsplit("_", 1)[1])

def execute_multi_turn_func_call(
    func_call_list: list[str],  # a list of strings of func calls
    initial_config: dict,
    involved_classes: list,
    model_name: str,
    test_entry_id: str,
    long_context: bool = False,
    is_evaL_run: bool = False,
) -> tuple[list[str], dict]:
    """
    Execute a list of function calls in a multi-turn scenario.
    Returns a tuple of (execution_results, involved_instances).
    """
    if is_evaL_run:
        model_name += "_eval"

    class_method_name_mapping = {}
    involved_instances = {}
    for class_name in involved_classes:
        module_file = CLASS_FILE_MAPPING[class_name]
        module_path = FUNC_SOURCE_DIR / f"{module_file}.py"
        
        # Create a spec from the file path
        spec = importlib.util.spec_from_file_location(module_file, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the class from the module
        class_obj = getattr(module, class_name)
        
        # Create an instance of the class
        if class_name in STATELESS_CLASSES:
            # For stateless classes, we don't need to pass any initial configuration
            instance = class_obj()
        else:
            # For stateful classes, create instance without initial_config and then load scenario
            instance = class_obj()
            if hasattr(instance, '_load_scenario'):
                instance._load_scenario(initial_config, long_context)
        
        # Add the instance to the global namespace
        instance_name = f"{model_name}_{test_entry_id}_{class_name.lower()}_instance"
        globals()[instance_name] = instance
        involved_instances[class_name] = instance
        
        # Get all methods of the class
        methods = inspect.getmembers(class_obj, inspect.isfunction)
        for method_name, method_obj in methods:
            if not method_name.startswith("_"):  # Skip private methods
                class_method_name_mapping[method_name] = instance_name

    execution_results = []
    for func_call in func_call_list:
        # Add the instance name to the method calls
        func_call = _process_method_call(func_call, class_method_name_mapping)

        # Fix leading zeros in the entire function call string before eval
        def fix_leading_zeros_in_string(match):
            number = match.group(0)
            # If it's a number with leading zeros and not 0, convert to decimal
            if number.startswith('0') and len(number) > 1 and not number.startswith('0x') and not number.startswith('0o') and not number.startswith('0b'):
                # Remove leading zeros and convert to int, then back to string
                return str(int(number))
            return number

        # Pattern to match numeric literals with potential leading zeros
        number_pattern = r'\b0\d+\b'
        func_call = re.sub(number_pattern, fix_leading_zeros_in_string, func_call)

        # Evaluate the function call
        try:
            # We need to make a copy here because otherwise the `eval(func_call)` would error. 
            func_call_copy = func_call
            # Before calling `eval`, we need to make sure that the function call is safe
            # We do so by checking if the function is `kill` or `exit`, etc.
            # Extract the function name first
            if "(" in func_call_copy:
                func_call_copy = func_call_copy.split("(")[0]
            # Situation where the function call is a method call
            if "." in func_call_copy:
                func_call_copy = func_call_copy.split(".")[1]
            if func_call_copy in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
                raise Exception(f"Function call {func_call_copy} is not allowed.")

            # Create a safe execution environment with necessary variables
            exec_globals = globals().copy()
            
            func_call_result = eval(func_call, exec_globals)

            if type(func_call_result) == str:
                pass
            elif type(func_call_result) == dict:
                # Some function returns a object instance, which is not serializable
                try:
                    func_call_result = json.dumps(func_call_result)
                except:
                    func_call_result = str(func_call_result)
            else:
                func_call_result = str(func_call_result)

            execution_results.append(func_call_result)
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")

    return execution_results, involved_instances

def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False

def _process_method_call(function_call_string: str, instance_mapping: dict) -> str:
    """
    Prepends the instance name to the function name for each of the function name represented in the string, you will
    also be provided with the mapping of method name to instance name.

    Example input:
    ```
    f(x = g((1, 2), h(3)), y = (4), z = (5, 6))
    ```

    Example return:
    ```
    a.f(x=a.g((1, 2), a.h(3)), y=(4), z=(5, 6))
    ```

    Args:
        function_call_string (str): The function call string to parse.
        instance_mapping (dict): A dictionary mapping method names to instance names.

    Returns:
        str: The parsed function call string with instance names prepended to method names.
    """

    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name

    # Regular expression to match function names
    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"

    # Replace function names with their class-prepended versions
    processed_string = re.sub(pattern, replace_function, function_call_string)

    # Fix leading zeros in numeric literals to prevent octal interpretation
    # This handles cases like 08, 09 which are invalid in Python 3
    def fix_leading_zeros(match):
        number = match.group(0)
        # If it's a number with leading zeros and not 0, convert to decimal
        if number.startswith('0') and len(number) > 1 and not number.startswith('0x') and not number.startswith('0o') and not number.startswith('0b'):
            # Remove leading zeros and convert to int, then back to string
            return str(int(number))
        return number

    # Pattern to match numeric literals with potential leading zeros
    number_pattern = r'\b0\d+\b'
    processed_string = re.sub(number_pattern, fix_leading_zeros, processed_string)

    return processed_string
