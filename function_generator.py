# The purpose of this python file is to generate the function based on user query that can be used for reusable collabroative data analytics

import importlib.util
import sys
import inspect
from pathlib import Path

from openai import OpenAI
from config import OPENAI_API_KEY

def api_call(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)

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


def construct_query(user_query: str) -> str:
    query = "Hello, you are an expert-level python programmer and tester. \
    Now, your task is to generate one or multiple function(s) that can be reused for modular collabroative data analytics. \
    What that means is your function must be an independent function that can be used by various users to perform data analytics. \
    It should be able to take in a dataset and perform some data analytics on it. \
    But it should not be specific to any dataset. \
    For example, if you are trying to calculate the mean of a column in a dataset, you should not hardcode the column name inside the function body. \
    Instead, you should pass the column name as an argument to the function. \
    In addition, since we are promoting modularity, the function itself should be a standalone function that doesn't rely on other functions, \
    except for built-in python functions or those from some modern python libraries. \
    You are free to decide what params you want to use. \
    You must follow the best practices of python programming and testing. \
    Such as using type hints, docstrings. It is also advised to add error handling. \
    Your final return should be the function solely that the user can copy and paste into their code. \
    You must also include any imports that are necessary for the function to work. \
    You can also include any comments that you think are necessary. \
    You can also include any additional information that you think is necessary. \
    You must not add things like backticks to declare code blocks as that will prevent the user from copying and pasting the code. \
    You must also not include any code that is not necessary for the function to work. \
    You must not include any conversational language in the function or at the start or end of the functions like 'Above are your functions'. \
    Or any illustrations or examples. \
    I will provide you with some examples that you can use as a reference."

    # Specify the path to your Python file
    module_path = Path("functionspace/m-data_preprocess.py")
    module_name = "m_data_preprocess"  # Use a Python-friendly alias for the module name

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    m_data_preprocess = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = m_data_preprocess
    spec.loader.exec_module(m_data_preprocess)

    # Now you have access to the functions
    df_keep_rows_by_index = m_data_preprocess.df_keep_rows_by_index
    df_keep_columns = m_data_preprocess.df_keep_columns
    df_delete_columns = m_data_preprocess.df_delete_columns
    df_rename_columns = m_data_preprocess.df_rename_columns

    # convert the function to string, also provide enumuration for the examples
    for index, func in enumerate([df_keep_rows_by_index, df_keep_columns, df_delete_columns, df_rename_columns]):
        flag, func_source = function_to_string(func)
        if flag:
            query += f"\n\nExample {index+1}:\n{func_source}\n\n"

    query += "Above are some examples that you can use as a reference. \
    But also note that these functions may not be that perfect as they are just for you to understand what modular reusable functions look like."

    query += "Based on the above information, now here's the user query: \n"
    query += user_query
    query += "\n\nEnd of user query. \
    Do note that the user may be asking for one or multiple functions. \
    So you must decide how many functions you want to generate based on the user query. \
    Also note that the input query may be codes already written by the user. \
    The codes may be written in R or Python or any other language. \
    If the query are codes, you must follow the exact same format and structure of the codes. \
    Including the indentation and the comments and also function names. Nothing should be modified if codes are provided already. \
    Again, you must not add anything that may cause syntax error such as backticks or any conversational language. \
    You may start generating the function(s) now."

    return query

# Function to convert the imported function to its string representation
def function_to_string(func):
    try:
        # Get the source code of the function
        func_source = inspect.getsource(func)
        flag = 1
        return flag, func_source
    except OSError:
        flag = 0
        return flag, f"Source code not available for {func.__name__}"


def generate_function(query):
    # Generate the function based on the user query
    function = api_call(construct_query(query))

    return function
