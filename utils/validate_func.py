import inspect
import importlib.util
import ast
from typing import Callable, List
# from config import *
import json

from openai import OpenAI

# create a global variable to store the api key
global_api_key = None


class ValidationResult:
    def __init__(
        self, function_name: str, validation_result: bool, message: str, issue_type: str
    ):
        self.function_name = function_name
        self.validation_result = validation_result
        self.message = message
        self.issue_type = issue_type
        if self.issue_type not in ["ERROR", "WARNING"]:
            raise ValueError("Invalid type value. Type must be 'ERROR' or 'WARNING'.")

    def __str__(self):
        return f"{self.issue_type}\nFunction: {self.function_name}\nValidation Result: {self.validation_result}\nMessage: {self.message}\n\n"

    def get_result(self) -> bool:
        return self.validation_result


class ValidationModel:
    def __init__(
        self, function: Callable, priority: int, description: str, issue_type: str
    ):
        self.function = function
        self.priority = priority
        self.description = description
        self.issue_type = issue_type
        if self.issue_type not in ["ERROR", "WARNING"]:
            raise ValueError("Invalid type value. Type must be 'ERROR' or 'WARNING'.")


def api_call(prompt):
    client = OpenAI(api_key=global_api_key)

    # Create a chat completion request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert-level python programmer and tester.",
            },
            {"role": "user", "content": f"{prompt}"},
        ],
    )

    reply = response.choices[0].message.content

    print(response.choices[0].message.content)

    return reply


def correction_proposal_gpt(name, obj, validation_result: List[ValidationResult]):
    prompt = "You are now a software engineer and your task is to propose a potential correction for the function that is provided to you. \
    You are given a function that is written in Python and the function contains certain issues that need to be corrected. \
    Based on the issue types and messages that are provided, you need to propose potential corrections for the functions. \
    You will be provided with the function source code and the issue type and message that you need to address. \
    Your final return response should be the function that contains no issue with the potential fix applied so that the user can just copy and paste directly. \
    You must not add any additional text or characters to the response. \
    You also must not add things like backtick and you must not explicitly declare python as the first line. \
    From here, I will provide you with the function source code and the issue type and message that you need to address. \
    You must aim to address every error message provided. For example, if it requires a default value, provide a dummy default value. \
    Keep in mind any dummy value must match its data type and should not be values like null or None or NA. \
    You must not change the function name unless an error message specifically asks for it. \
    You should also aim to address the warning messages provided. They may not be critical but they are good practices to follow. \
    But if addressing warning messages would require a significant change in the function or may modify the function logic, you should skip them."

    prompt += f"\n\nFunction: {name}\nSource Code: {inspect.getsource(obj)}\n"
    for result in validation_result:
        prompt += f"\n\nIssue Type: {result.issue_type}\nMessage: {result.message}\n"

    prompt += "\n\nFor your reference, these issue messages are generated based on the following pre-defined rules, "
    for rule in validation_rules:
        prompt += f"\nDescription: {rule.description}\nIssue Type: {rule.issue_type}\n"

    response = api_call(prompt)

    return response


# Warnings -------------------------------------------------------------------------------
# Check for function relevance
warning_response_cache = {}


def warning_api_call(name, obj):
    if name in warning_response_cache:
        return warning_response_cache[name]

    prompt = "You are now a software engineer and your task is to assess the relevance of the functions that are provided to you. \
    You are given a function that is written in Python and you need to determine whether the function is relevant to the existing functions (docstrings of existing functions will be provided) and/or related to the finance domain. \
    In addition, you should also determine whether the function is meant to have the input and output parameters. \
    What that means is based on the actual function context, it may be a function that does webscraping thus does have the output path (to store webscraping result) but not the input path. \
    Or it may be a function that does not have the output path but does have the input path. \
    So your final return response should include the result and must strictly follow this format 'Relevance: [true/false]; Parameter: [input/output/both]' \
    You must not add any additional text or characters to the response. "
    # From here, I will provide you with the docstrings of existing functions that you need to assess."

    # iterate through the existing functions in func.py and provide the docstrings for the prompt
    # import full_func as func
    # for func_name, func_obj in inspect.getmembers(func):
    #     if inspect.isfunction(func_obj) and func_obj.__module__ == func.__name__:
    #         prompt += (
    #             f"\n\nFunction: {func_name}\nDocstring: {inspect.getdoc(func_obj)}\n"
    #         )

    # append the function (source_code) that needs to be assessed
    prompt += "Here is the function that you need to assess"
    prompt += f"\n\nFunction: {name}\nSource Code: {inspect.getsource(obj)}\n"

    response = api_call(prompt)
    warning_response_cache[name] = response

    return response


def check_relevance(name, obj):
    response = warning_api_call(name, obj)

    if "Relevance: true" in response:
        return ValidationResult(
            name,
            True,
            "Function is relevant to existing functions and related to finance domain",
            "WARNING",
        )
    else:
        return ValidationResult(
            name,
            False,
            "Function is not relevant to existing functions or related to finance domain",
            "WARNING",
        )


def check_input_output_stream(name, obj):
    sig = inspect.signature(obj)
    parameters = sig.parameters

    has_input_stream = False
    has_output_stream = False
    for param in parameters.values():
        if param.annotation == str and param.default != inspect.Parameter.default:
            if "input" in param.name:
                has_input_stream = True
            if "output" in param.name:
                has_output_stream = True

    response = warning_api_call(name, obj)

    if "Parameter: both" in response:
        if not has_input_stream or not has_output_stream:
            return ValidationResult(
                name,
                False,
                "Function is expected to contain both input and output stream parameters",
                "WARNING",
            )
    elif "Parameter: input" in response:
        if not has_input_stream:
            return ValidationResult(
                name,
                False,
                "Function is expected to contain input stream parameter",
                "WARNING",
            )
    elif "Parameter: output" in response:
        if not has_output_stream:
            return ValidationResult(
                name,
                False,
                "Function is expected to contain output stream parameter",
                "WARNING",
            )

    return ValidationResult(
        name, True, "Function has input and output stream", "WARNING"
    )


# Errors -------------------------------------------------------------------------------
def check_docstring(name, obj):
    if inspect.getdoc(obj) is None:
        return ValidationResult(
            name, False, "Function does not have a docstring", "ERROR"
        )

    # TODO: Check whether the inspect.getdoc(obj) matches the function's actual behavior
    prompt = "You are now a software engineer and your task is to assess the docstring of the function that is provided to you. \
    You are given a function that is written in Python and you need to determine whether the function contains a docstring that accurately describes the function's behavior. \
    Based on the function's source code, you need to assess whether the docstring is accurate. \
    Your final return response should only be True or False and nothing else. \
    From here, I will provide you with the function source code and the docstring that you need to assess."

    prompt += f"\n\nFunction: {name}\nSource Code: {inspect.getsource(obj)}\n"
    prompt += f"\n\nDocstring: {inspect.getdoc(obj)}\n"

    response = api_call(prompt)

    if response == "False":
        return ValidationResult(
            name,
            False,
            "Function docString does not accurately describe the function behavior",
            "WARNING",
        )
    else:
        return ValidationResult(name, True, "Function docString is valid", "ERROR")


def check_annotations_and_default_values(name, obj):
    sig = inspect.signature(obj)
    parameters = sig.parameters

    validation_result_list = []

    for param in parameters.values():
        if param.annotation == inspect.Parameter.empty:
            validation_result_list.append(
                ValidationResult(
                    name,
                    False,
                    f"Parameter {param.name} does not have annotation",
                    "ERROR",
                )
            )
        if param.default == inspect.Parameter.empty:
            validation_result_list.append(
                ValidationResult(
                    name,
                    False,
                    f"Parameter {param.name} does not have default value",
                    "ERROR",
                )
            )

    if len(validation_result_list) > 0:
        return validation_result_list
    else:
        return ValidationResult(
            name, True, "Function has annotations and default values", "ERROR"
        )


# Below doesn't seem to be useful, the bokeh library seems to be auto rejecting any file containing the syntax error
# Though the error message is not shown on our interface
def check_syntax_error(name, obj):
    try:
        ast.parse(inspect.getsource(obj))
    except SyntaxError as e:
        return ValidationResult(
            name,
            False,
            f"Syntax Error on line {e.lineno} character {e.offset}: {e.msg}\nCode: {e.text}",
            "ERROR",
        )

    return ValidationResult(name, True, "Function has no syntax error", "ERROR")


def check_duplicate_function(name, obj):
    full_mapping_file_path = "full_func.json"

    with open(full_mapping_file_path, "r") as f:
        full_mapping = json.load(f)

    # full mapping is a dictionary with key as file name and value as list of function names
    for _, func_list in full_mapping.items():
        if name in func_list:
            return ValidationResult(
                name,
                False,
                "Function name already exists in the current functionspace",
                "ERROR",
            )

    return ValidationResult(name, True, "Function name is unique", "ERROR")


# Create a sorted list of validation models
validation_rules = [
    ValidationModel(
        check_syntax_error, 1, "Function must not contain syntax error", "ERROR"
    ),
    ValidationModel(check_docstring, 2, "Function must contains docString", "ERROR"),
    ValidationModel(
        check_annotations_and_default_values,
        3,
        "Function parameters must contain both type annotation and default values",
        "ERROR",
    ),
    ValidationModel(
        check_relevance,
        4,
        "Ideally, function should be relevant to existing functions or related to finance domain",
        "WARNING",
    ),
    ValidationModel(
        check_input_output_stream,
        5,
        "Ideally, function should have both input and output streams",
        "WARNING",
    ),
]

validation_rules.sort(key=lambda x: x.priority)


def insert_validation_rule_in_place(rules: list[ValidationModel], new_rule: ValidationModel):
    # Shift the priority of existing rules in place
    for rule in rules:
        if rule.priority >= new_rule.priority:
            rule.priority += 1

    
    rules.append(new_rule)

    # Sort in-place to maintain the order by priority
    rules.sort(key=lambda x: x.priority)

    return rules


validation_rules = insert_validation_rule_in_place(
    validation_rules,
    ValidationModel(
        check_duplicate_function, 4, "Function name must be unique", "ERROR"
    ),
)


def validate_func(module):
    final_validate_result = {}
    all_proposed_correction = {}

    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            print(f"Function: {name}")
            # print(f'Args: {inspect.getfullargspec(obj)}')
            # print(f'Doc: {inspect.getdoc(obj)}')
            # print(f'Annotations: {obj.__annotations__}')
            print()

            validate_result = []

            for validation_model in validation_rules:
                result = validation_model.function(name, obj)

                print(f"Processing function {name} with validation '{validation_model.description}' and priority {validation_model.priority}")

                if isinstance(result, list):
                    for res in result:
                        print("Temporarily Result (list) ",res)
                        validate_result.append(res)
                else:
                    print("Temporarily Result ",result)
                    validate_result.append(result)

            final_validate_result[name] = validate_result

            correction_needed = False

            for result in validate_result:
                print("Result ",result)
                if result.get_result() is False and result.issue_type == "ERROR":
                    correction_needed = True

            if correction_needed:
                correction = correction_proposal_gpt(name, obj, validate_result)
                all_proposed_correction[name] = correction
            else:
                all_proposed_correction[name] = inspect.getsource(obj)

    return final_validate_result, all_proposed_correction


def load_module(file_path):
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_validation_result(file_path, api_key):
    module = load_module(file_path)

    #assign the api key to the global variable
    global global_api_key
    global_api_key = api_key

    validation_result, proposed_correction = validate_func(module)
    flag = True
    message, error_message, warning_message = "", "", ""
    for _, results in validation_result.items():
        for result in results:
            if result.issue_type == "ERROR" and result.validation_result == False:
                error_message += str(result)
                flag = False
            elif result.issue_type == "WARNING":
                warning_message += str(result)

    for name, correction in proposed_correction.items():
        message += f"Function: {name}\nProposed Correction or Original function: \n{correction}\n\n"

    message += "----------------------------------------------------------------------------------------------------------------------\n\n"
    message += "Below are the validation result which may help you to understand the proposed correction and the reasoning.\n\n"
    message += error_message + "\n" + warning_message + "\n\n"

    if flag:
        message = (
            "All functions are valid"
            if warning_message == ""
            else "Some warning messages that may be useful for further improvements\n\n"
            + warning_message
        )
    # print(flag, message)
    return flag, message, proposed_correction


# print(function_validation_result('func_api.py'))


def extract_import_statements(file_path):
    with open(file_path, 'r') as file:
        # Parse the file content into an AST
        tree = ast.parse(file.read(), filename=file_path)

        # List to hold the import statements
        import_statements = []

        # Walk through the nodes of the AST
        for node in ast.walk(tree):
            # Check for 'import' statements like 'import module_name'
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Check if the import has an alias (e.g., 'import numpy as np')
                    if alias.asname:
                        import_statements.append(f"import {alias.name} as {alias.asname}")
                    else:
                        import_statements.append(f"import {alias.name}")

            # Check for 'from ... import ...' statements
            elif isinstance(node, ast.ImportFrom):
                # Get module name and imported names
                module = node.module or ''
                for alias in node.names:
                    # Handle aliases in 'from ... import ... as ...' statements
                    if alias.asname:
                        import_statements.append(f"from {module} import {alias.name} as {alias.asname}")
                    else:
                        import_statements.append(f"from {module} import {alias.name}")
        
        return import_statements