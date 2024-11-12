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
    top_k = 5

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
    conn.close()

    system_message = f'''You are an data analytics expert that recommends pipelines based on user queries. 
                        You have access to a knowledge base of pipelines and their components. 
                        You need to generate a pipeline based on the user query and JSON configuration provided. 
                        You also have access to a list of functions in the knowledge base that can be used in the pipeline.
                        Here are some functions in the knowledge base:{functions_kb}
                        '''

    user_message = (f"Recommend a pipeline based on both user_query and user_pipeline."
                    f"The user query is: {user_query}."
                    f"The user pipeline is: {user_pipeline}."
                    f"Here are the functions available in the knowledge base: {functions_kb}, do not generate something you cannot find here."
                    f"Ensure the structure and style are consistent with the existing functions."
                    f"The final outcome should be a JSON configuration of the pipeline same as the format  {prompt_format}"
                    f"Give the reason of summary to explain why the pipeline and special parameters are recommended."
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

    
    system_message = f'''You are an data analytics expert that recommends paramters for the analytics pipeline based on user queries. 
                        You need to generate a parameters based on the user query and JSON pipline provided. 
                        '''

    user_message = (f"Recommend a parameters based on the given analytics pipeline details- keys are functions, values are parameters."
                    f"The analytics pipeline is: {generated_pipeline}."
                    f"Here is thet task that user wanna do: {user_query}."
                    f"Do not change the given pipeline, only suggest the parameters."
                    f"The final outcome should be the same pipeline with the parameters you suggested."
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
