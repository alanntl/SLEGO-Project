import os
import ast
import panel as pn
import inspect
import json
def test_function(input_string:str='Hello!', 
          output_file_path:str='dataspace/output.txt'):
    """
    A simple function to save the provided input string to a specified text file and return the string.

    Parameters:
    - input_string (str): The string to be saved.
    - output_file_path (str): The file path where the string should be saved.

    Returns:
    - str: The same input string.
    """

    # Open the file at the specified path in write mode
    with open(output_file_path, 'w') as file:
        # Write the input string to the file
        file.write(input_string)

    # Return the input stringƒ√ƒ√
    
    return input_string

def _compute(module_name, input):
    module = __import__(module_name)

    #pipeline_dict = ast.literal_eval(input)
    pipeline_dict = json.loads(text)
    output = ""
    for function_name, parameters in pipeline_dict.items():
        function = eval(f"module.{function_name}")
        result = function(**parameters)

        output += "\n===================="+function_name+"====================\n\n"
        output += str(result)

    return output

def _create_multi_select_combobox(target_module):
    """
    Creates a multi-select combobox with all functions from the target_module.
    """
    
    # Get the module name (e.g., "func" if your module is named func.py)
    module_name = target_module.__name__
    
    # Get a list of all functions defined in target_module
    functions = [name for name, obj in inspect.getmembers(target_module, inspect.isfunction)
                 if obj.__module__ == module_name and not name.startswith('_')]

    # Create a multi-select combobox using the list of functions
    multi_combobox = pn.widgets.MultiChoice(name='Functions:', options=functions, height=150)

    return multi_combobox


# def _create_multi_select_combobox(func):
#   """
#   Creates a multi-select combobox with all functions from the func.py file.
#   """

#   # Get a list of all functions in the func.py file.
#   functions = [name for name, obj in inspect.getmembers(func)
#                 if inspect.isfunction(obj) and not name.startswith('_')]

#   # Create a multi-select combobox using the list of functions.
#   multi_combobox = pn.widgets.MultiChoice(name='Functions:', options=functions,  height=150)

#   return multi_combobox


def _extract_parameter(func):
    """
    Extracts the names and default values of the parameters of a function as a dictionary.

    Args:
        func: The function to extract parameter names and default values from.

    Returns:
        A dictionary where the keys are parameter names and the values are the default values.
    """
    signature = inspect.signature(func)
    parameters = signature.parameters

    parameter_dict = {}
    for name, param in parameters.items():
        if param.default != inspect.Parameter.empty:
            parameter_dict[name] = param.default
        else:
            parameter_dict[name] = None

    return parameter_dict


import os
import json

def __combine_json_files(directory, output_file):
    """
    Combine all JSON files in a directory into a single JSON file.

    Args:
    directory (str): The directory containing JSON files.
    output_file (str): The path to the output JSON file.
    """
    combined_data = []  # List to hold data from all JSON files

    # Loop over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):  # Check for JSON files
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)  # Load JSON data from file
                combined_data.append(data)  # Append data to the list

    # Write combined data to output JSON file
    with open(output_file, 'w') as file:
        json.dump(combined_data, file, indent=4)  # Use 'indent' for pretty-printing

    print("All JSON files have been combined into:", output_file)

# # Example usage:
# combine_json_files('/content/drive/MyDrive/SLEGO/slegospace/knowledgespace', '/content/drive/MyDrive/SLEGO/slegospace/knowledgespace/kownledge.json')

import os
import rdflib
import networkx as nx
from pyvis.network import Network
import webbrowser

# Step 1: Define the style
style = {
    "color": "gray",
    "shape": "ellipse",
    "size": 10
}

# Step 2: Define the construct_subgraph function
def __construct_subgraph(file_path, query, output_file):
    """
    Executes a CONSTRUCT query on an RDF graph and saves the resulting subgraph to a file.

    Args:
        file_path (str): Path to the input RDF Turtle file.
        query (str): SPARQL CONSTRUCT query string.
        output_file (str): Path to the output Turtle file.
    """
    # Load the original graph
    g = rdflib.Graph()
    g.parse(file_path, format='turtle')

    # Execute the CONSTRUCT query
    subgraph = g.query(query)

    # Get the resulting graph
    subgraph_graph = rdflib.Graph()
    for triple in subgraph:
        subgraph_graph.add(triple)

    # Save the subgraph to a Turtle file
    subgraph_graph.serialize(destination=output_file, format='turtle')

    print(f"Subgraph saved to {output_file}")

# Step 3: Convert the Graph to Results Format
def __graph_to_results(graph):
    results = []
    for subj, pred, obj in graph:
        row = [('subject', subj), ('predicate', pred), ('object', obj)]
        results.append(row)
    return results

# Step 4: Helper Function to Extract Local Names
def __get_local_name(uri):
    if isinstance(uri, rdflib.term.URIRef):
        uri = str(uri)
    if '#' in uri:
        return uri.split('#')[-1]
    elif '/' in uri:
        return uri.rstrip('/').split('/')[-1]
    else:
        return uri

# Step 5: Visualization Function
def __visualize_query_results_interactive(results):
    G = nx.DiGraph()
    node_types = {}

    # Build the graph
    for row in results:
        row_dict = dict(row)
        subj = __get_local_name(row_dict['subject'])
        pred = row_dict['predicate']
        obj = __get_local_name(row_dict['object'])
        
        # Capture rdf:type relationships to identify node types
        if str(pred) == rdflib.RDF.type:
            node_types[subj] = obj

        G.add_edge(subj, obj, label=__get_local_name(pred))

    # Initialize the Network object
    net = Network(
        notebook=True, height="1000px", width="100%",
        bgcolor="#ffffff", font_color="black", directed=True,
        cdn_resources='remote'
    )

    # Set visualization options and incorporate the style
    net.set_options(f"""
    var options = {{
        "nodes": {{
            "shape": "{style['shape']}",
            "color": "{style['color']}",
            "size": {style['size']},
            "font": {{
                "size": 14,
                "face": "Tahoma"
            }}
        }},
        "edges": {{
            "arrows": {{
                "to": {{
                    "enabled": true,
                    "scaleFactor": 1
                }}
            }},
            "smooth": {{
                "type": "continuous"
            }}
        }},
        "layout": {{
            "hierarchical": {{
                "enabled": true,
                "levelSeparation": 250,
                "nodeSpacing": 200,
                "treeSpacing": 300,
                "blockShifting": true,
                "edgeMinimization": true,
                "parentCentralization": true,
                "direction": "LR",
                "sortMethod": "hubsize"
            }}
        }},
        "physics": {{
            "enabled": false
        }}
    }}
    """)

    # Add nodes to the network
    for node in G.nodes():
        node_type_uri = node_types.get(node, None)
        node_type = __get_local_name(node_type_uri) if node_type_uri else 'Unknown'
        net.add_node(node, label=node, title=f"Type: {node_type}", **style)

    # Add edges to the network
    for u, v, data in G.edges(data=True):
        label = data.get('label', '')
        net.add_edge(u, v, label=label, title=label)

    # Generate and show the network
    output_html = 'slegospace/ontologyspace/interactive_graph.html'
    net.show(output_html)
    webbrowser.open('file://' + os.path.realpath(output_html))
    print(f"Visualization saved to {output_html} and opened in your default browser.")

def __extract_and_visualize_subgraph(file_path, subgraph_file, query):
    """
    Extracts and visualizes a subgraph based on a SPARQL CONSTRUCT query.

    Args:
        file_path (str): Path to the input RDF Turtle file.
        subgraph_file (str): Path to the output Turtle file where the subgraph will be saved.
        query (str): SPARQL CONSTRUCT query string to extract the subgraph.
    """
    # Construct and save the subgraph
    g = rdflib.Graph()
    g.parse(file_path, format='turtle')
    subgraph = g.query(query)
    subgraph_graph = rdflib.Graph()
    for triple in subgraph:
        subgraph_graph.add(triple)
    subgraph_graph.serialize(destination=subgraph_file, format='turtle')

    print(f"Subgraph saved to {subgraph_file}")

    # Parse the subgraph
    g = rdflib.Graph()
    g.parse(subgraph_file, format='turtle')

    print(f"Parsed {len(g)} triples from the subgraph.")

    # Convert the subgraph to results
    results = __graph_to_results(g)

    # Visualize the subgraph
    __visualize_query_results_interactive(results)