import json
import requests
import sqlite3
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import openai
from openai import OpenAI


def pipeline_recommendation(db_path,user_query,user_pipeline, openai_api_key):
    
    client = OpenAI(api_key=openai_api_key)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()       
    cursor.execute("SELECT pipeline_name, description, pipeline_details, embedding FROM pipelines")

    pipelines_info = cursor.fetchall()
    
    all_queries = user_query + ' ' +str(user_pipeline)

    response = client.embeddings.create(input=all_queries, model="text-embedding-ada-002")
    # query_embedding_json = json.dumps(response.data[0].embedding)

    # query_embedding = np.array(json.loads(query_embedding_json))

    query_embedding = np.array(response.data[0].embedding)
    top_k = 10

    similarities = []
    for pipeline_name, description, microservices_details, embedding_str in pipelines_info:
        embedding = np.array(json.loads(embedding_str))
        similarity_score = 1 - cosine(query_embedding, embedding)
        similarity_percentage = similarity_score * 100
        similarities.append({
            'pipeline_name': pipeline_name,
            'description': description,
            'pipeline_details': json.loads(microservices_details),
            'similarity_percentage': similarity_percentage
        })

    similarities.sort(key=lambda x: x['similarity_percentage'], reverse=True)
    top_pipelines = similarities[:top_k]
    top_pipelines_df = pd.DataFrame(top_pipelines)

    components_description_list = []
    components_source_code_list = []

    for pipeline in top_pipelines_df['pipeline_details'].values:
        component_description = {}
        component_source_code = {}
        for component in pipeline:
            if isinstance(component, dict):
                component_name = list(component.keys())[0]
            elif isinstance(component, str):
                component_name = component
            else:
                continue  # Skip if component is neither dict nor string

            cursor.execute("SELECT module_name, docstring, source_code FROM functions WHERE component_name = ?", (component_name,))
            component_info = cursor.fetchall()
            
            if component_info:
                component_description[component_name] = component_info[0][1]
                component_source_code[component_name] = component_info[0][2]
            else:
                component_description[component_name] = 'No description found for this component'
                component_source_code[component_name] = 'No source code found for this component'           

        components_description_list.append(component_description)
        components_source_code_list.append(component_source_code)

        prompt_format ='''
                            {
                                "function1": {
                                    "param1": "default value 1",
                                    "param2":"default value 2",
                                    "param3":"default value 3",
                                    },

                                "function2": {
                                    "param1": "default value 1",
                                    "param2":"default value 2",
                                },
                                }
                                '''
        

    functions_kb = str(top_pipelines_df.to_dict())
    #functions_kb = str(all_components)
    conn.close()

    # system_message = f'''You are an data analytics expert that recommends pipelines based on user queries. 
    #                     You have access to a knowledge base of pipelines and their components. 
    #                     You need to generate a pipeline based on the user query and JSON configuration provided. 
    #                     You also have access to a list of functions in the knowledge base that can be used in the pipeline.
    #                     Here are some functions in the knowledge base:{functions_kb}
    #                     '''
    system_message = f'''You are a data analytics expert specializing in recommending analytics pipelines based on user needs.
                        You have a knowledge base of pipelines and components, with specific JSON configurations available.
                        When a user provides a query, suggest the most relevant pipeline configuration by modifying the parameters to best fit the user’s needs. 
                        Only recommend components that exist within the knowledge base and ensure that the output JSON format is consistent with existing examples.
                        However, you can use all components cross different pipelines.
                        Example JSON format for output:
                        {{
                            "function_name": {{
                                "parameter1": "value1",
                                "parameter2": "value2",
                                ...
                            }},
                            ...
                        }}
                        '''

    # user_message = (f"Recommend a pipeline based on both user_query and user_pipeline."
    #                 f"The user query is: {user_query}."
    #                 f"The user pipeline is: {user_pipeline}."
    #                 f"Here are the functions available in the knowledge base: {functions_kb}, do not generate something you cannot find here."
    #                 f"Ensure the structure and style are consistent with the existing functions."
    #                 f"The final outcome should be a JSON configuration of the pipeline same as the format  {prompt_format}"
    #                 f"Give the reason of summary to explain why the pipeline and special parameters are recommended."
    #                 )
    user_message = (f"Based on the user's needs, recommend the best analytics pipeline from the knowledge base by adapting parameters."
                    f"User Query: {user_query}."
                    f"Existing Pipeline: {user_pipeline}."
                    f"Available functions in the knowledge base: {functions_kb}."
                    f"Make sure to use only components from the knowledge base and keep the output in the same JSON format."
                    f"Explain why each recommended pipeline component is a good fit and provide any relevant details for parameter adaptation."
                    f"Output format should follow this example JSON format: {prompt_format}"
                    f"actually you are encoruage to grab the functions from different pipelines, but make sure the whole pipelien make sense and not doing many different things."
                    f"dont give comment inside the json, becuase users cannot copy and paste the json into the program"
                    )

    response = client.chat.completions.create(
        model="gpt-4o",
        #response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=1,
        max_tokens=1280,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        )
    response_text = response.choices[0].message.content.strip() # response['choices'][0]['message']['content'].strip()
    return response_text


def pipeline_parameters_recommendation(user_query, generated_pipeline, openai_api_key):
    client = OpenAI(api_key=openai_api_key)

    
    # system_message = f'''You are an data analytics expert that recommends paramters for the analytics pipeline based on user queries. 
    #                     You need to generate a parameters based on the user query and JSON pipline provided. 
    #                     '''
    system_message = f'''You are a data analytics expert providing parameter suggestions for existing analytics pipelines.
                        Use the user’s query to adjust parameters within a given JSON pipeline structure to enhance the pipeline’s effectiveness for the specified task.
                        Retain the original pipeline structure; modify only the parameter values as needed to tailor the pipeline to the user’s needs.
                        '''

    # user_message = (f"Recommend a parameters based on the given analytics pipeline details- keys are functions, values are parameters."
    #                 f"The analytics pipeline is: {generated_pipeline}."
    #                 f"Here is thet task that user wanna do: {user_query}."
    #                 f"Do not change the given pipeline, only suggest the parameters."
    #                 f"The final outcome should be the same pipeline with the parameters you suggested."
    #                 )
    user_message = (f"Recommend parameters for the analytics pipeline based on the given task and pipeline structure."
                    f"Task Description: {user_query}."
                    f"Pipeline Structure: {generated_pipeline}."
                    f"Retain the given structure, making modifications only to the parameter values."
                    f"Ensure the parameters align with the user’s task while staying true to the pipeline's intended functionality."
                    f"rename the parameters of input and ouput file path name match the user's task."
                    f"dont give comment inside the json, becuase users cannot copy and paste the json into the program"
                )

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=1,
        max_tokens=1280,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        )
    response_text = response.choices[0].message.content.strip() # response['choices'][0]['message']['content'].strip()
    return response_text
