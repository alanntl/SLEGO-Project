import ast
import inspect
import os
import json
from typing import List, Dict

def validate_microservice(microservice_path: str) -> Dict[str, str]:
    """
    Validate a Python microservice script based on SLEGO guidelines.

    Parameters:
    microservice_path (str): Path to the microservice Python file.

    Returns:
    dict: Validation results, including success or errors.
    """
    validation_results = {"status": "success", "errors": []}

    try:
        # Read the Python file content
        with open(microservice_path, 'r') as file:
            source_code = file.read()

        # Parse the source code to an AST
        tree = ast.parse(source_code)

        # Ensure there is at least one function defined
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if not functions:
            validation_results["status"] = "error"
            validation_results["errors"].append("No function defined in the microservice.")
            return validation_results

        for function in functions:
            # Check for a docstring
            if not ast.get_docstring(function):
                validation_results["errors"].append(
                    f"Function '{function.name}' is missing a docstring."
                )

            # Validate parameters
            args = [arg.arg for arg in function.args.args]
            if not args:
                validation_results["errors"].append(
                    f"Function '{function.name}' has no input parameters."
                )



        # Ensure standardized file naming
        if not os.path.basename(microservice_path).startswith("m-"):
            validation_results["errors"].append("File name does not follow the 'm-' prefix convention.")

        # Ensure standardized function naming


        # # Check for descriptive docstring and standard input/output paths
        # for function in functions:
        #     docstring = ast.get_docstring(function)
        #     if docstring and "input_file_path" not in docstring:
        #         validation_results["errors"].append(
        #             f"Function '{function.name}' docstring should describe 'input_file_path'."
        #         )

        #     if docstring and "output_file_path" not in docstring:
        #         validation_results["errors"].append(
        #             f"Function '{function.name}' docstring should describe 'output_file_path'."
        #         )

    except Exception as e:
        validation_results["status"] = "error"
        validation_results["errors"].append(str(e))

    if validation_results["errors"]:
        validation_results["status"] = "error"

    return validation_results

def validate_microservices_in_directory(directory_path: str) -> List[Dict[str, str]]:
    """
    Validate all Python microservice scripts in a directory.

    Parameters:
    directory_path (str): Path to the directory containing microservices.

    Returns:
    list: List of validation results for each microservice.
    """
    results = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".py"):
            file_path = os.path.join(directory_path, file_name)
            result = validate_microservice(file_path)
            results.append({"file": file_name, "result": result})
    return results

## Example usage
##directory_to_validate = "./microservices"
#validation_results = validate_microservices_in_directory(directory_to_validate)

# Output results
##for result in validation_results:
#    print(json.dumps(result, indent=4))
