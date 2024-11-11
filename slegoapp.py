import os
import sys
import subprocess
import platform
import logging
import json
import re
import time
import inspect
import itertools
import importlib
from datetime import datetime
from typing import Dict, Any
import ast
import rdflib
import asyncio
from IPython.display import Javascript, display

from rdflib import Graph, URIRef
from pyvis.network import Network
import logging
import webbrowser
import os
import panel as pn
import pandas as pd
import kglab
from pyvis.network import Network
from rdflib import URIRef
import networkx as nx
# Install required packages if they are not already installed

def check_and_install_packages():
    required_packages = ['panel', 'param', 'pandas', 'kglab', 'pyvis', 'rdflib']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

check_and_install_packages()


import param
import pandas as pd
import kglab
from pyvis.network import Network
from rdflib import URIRef

# Import recommender module
import recommender as rc
from validate_func import *
from function_generator import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Panel extensions
pn.extension('ace', 'jsoneditor', 'tabulator', sizing_mode='stretch_both')

class SLEGOApp:
    def __init__(self, config: Dict[str, Any]):
        logger.info("Initializing SLEGOApp...")
        self.config = config
        self.folder_path = config['folder_path']
        self.functionspace = config['functionspace']
        self.dataspace = config['dataspace']
        self.recordspace = config['recordspace']
        self.knowledgespace = config['knowledgespace']
        self.ontologyspace = config['ontologyspace']

        # Log configuration paths
        logger.info(f"Configuration paths:")
        logger.info(f"  folder_path: {self.folder_path}")
        logger.info(f"  functionspace: {self.functionspace}")
        logger.info(f"  dataspace: {self.dataspace}")
        logger.info(f"  recordspace: {self.recordspace}")
        logger.info(f"  knowledgespace: {self.knowledgespace}")
        logger.info(f"  ontologyspace: {self.ontologyspace}")

        self.initialize_widgets()
        self.setup_func_module()
        self.setup_event_handlers()
        self.create_layout()
        self.create_template()
        self.full_func_mapping_path = 'full_func.json'
        self.setup_full_func_mapping()
        self.output_text.value = '\nHello!\n\nWelcome to SLEGO - A Platform for Collaborative and Modular Data Analytics.\n\nPlease select the modules and functions to get started.\n\n'
        logger.info("SLEGOApp initialized.")

    def initialize_widgets(self):
        logger.info("Initializing widgets...")
        # Ensure functionspace exists
        if not os.path.exists(self.functionspace):
            os.makedirs(self.functionspace)
            logger.info(f"Created functionspace directory: {self.functionspace}")

        # Include all .py files in the functionspace directory
        self.py_files = [f for f in os.listdir(self.functionspace) if f.endswith('.py')]

        if not self.py_files:
            logger.warning("No .py modules found in the functionspace.")
        else:
            logger.info(f"Python files available for selection: {self.py_files}")

        # Set default selected modules
        default_modules = ['func_yfinance.py']
        default_selected = [module for module in default_modules if module in self.py_files]
        self.funcfilecombo = pn.widgets.MultiChoice(
            name='Select Module(s)',
            value=default_selected,
            options=self.py_files,
            height=100
        )
    # ... rest of the method remains the same ...

        self.compute_btn = pn.widgets.Button(name='Compute', height=50, button_type='primary')
        self.savepipe_btn = pn.widgets.Button(name='Save Pipeline', height=35)
        self.pipeline_text = pn.widgets.TextInput(value='', placeholder='Input Pipeline Name', height=35)
        self.json_toggle = pn.widgets.Toggle(name='Input mode: text or form', height=35, button_type='warning')
        self.json_editor = pn.widgets.JSONEditor(value={}, mode='form')
        self.json_text = pn.widgets.TextAreaInput(value='', placeholder='Input the parameters')
        self.input_text = pn.widgets.TextAreaInput(
            value='', 
            placeholder='User query inputs for recommendation or SPARQL or system messages:', 
            name='User input:', 
            height=150
        )
        self.output_text = pn.widgets.TextAreaInput(
            value='', 
            placeholder='Results will be shown here', 
            name='System output message:'
        )

        # Added missing widgets with specified heights
        self.recommendation_btn = pn.widgets.Button(
            name='Get Recommendation', 
            height=35, 
            button_type='success'
        )
        self.recomAPI_text = pn.widgets.TextInput(
            value='', 
            placeholder='Your AI API key', 
            height=35
        )

        self.func_generator_btn = pn.widgets.Button(name='Generate Function', height=35, button_type='success')

        # File management widgets
        self.folder_select = pn.widgets.Select(
            name='Select Folder',
            options=[item for item in os.listdir(self.folder_path) 
                     if os.path.isdir(os.path.join(self.folder_path, item))] + ['/'],
            value='dataspace',
            height=50
        )
        self.file_text = pn.widgets.TextInput(
            value='/dataspace', 
            placeholder='Input the file name', 
            height=35
        )
        self.filefolder_confirm_btn = pn.widgets.Button(name='Confirm', height=35)
        self.file_view = pn.widgets.Button(name='View', height=35)
        self.file_download = pn.widgets.Button(name='Download', height=35)
        self.file_upload = pn.widgets.Button(name='Upload', height=35)
        self.file_input = pn.widgets.FileInput(name='Upload file', height=35)
        self.file_delete = pn.widgets.Button(name='Delete', height=35)
        self.file_table = self.create_file_table()

        self.param_widget_tab = pn.Tabs(
            ('JSON Input', self.json_editor), 
            ('Text Input', self.json_text),
            scroll=True,
        )
        self.ontology_btn = pn.widgets.Button(name='Show Ontology', height=35)

        self.rules_popup = pn.Card(pn.pane.Markdown("## Modal Title\nThis is the content inside the modal."), title="Modal", width=80, height=80, header_background="lightgray")
        self.rules_button = pn.widgets.Button(name="", icon="info-circle", width=10, height=35, button_type="light")        

        # Placeholder for funccombo
        self.funccombo_pane = pn.Column()
        logger.info("Widgets initialized.")

    def setup_func_module(self):
        logger.info("Setting up func module...")
        selected_modules = self.funcfilecombo.value  # Get the selected modules
        if not selected_modules:
            logger.error("No modules selected. Please ensure desired modules exist in the functionspace.")
            return
        self.update_func_module(selected_modules)

    def create_file_table(self):
        logger.info("Creating file table...")
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        logger.info(f"Selected folder path: {selected_folder_path}")
        if os.path.exists(selected_folder_path):
            file_list = os.listdir(selected_folder_path)
            df_file = pd.DataFrame(file_list, columns=['Filter Files :'])
            logger.info(f"Files in {selected_folder_path}: {file_list}")
            return pn.widgets.Tabulator(df_file, header_filters=True, show_index=True)
        else:
            logger.warning(f"Folder {selected_folder_path} does not exist.")
            return pn.widgets.Tabulator(pd.DataFrame(), header_filters=True, show_index=True)

    def setup_event_handlers(self):
        logger.info("Setting up event handlers...")
        self.funcfilecombo.param.watch(self.funcfilecombo_change, 'value')
        # Remove old funccombo watcher if it exists
        if hasattr(self, 'funccombo'):
            self.funccombo.param.watch(self.funccombo_change, 'value')
        self.json_text.param.watch(self.json_text_change, 'value')
        self.json_toggle.param.watch(self.json_toggle_clicked, 'value')
        self.json_editor.param.watch(self.json_editor_change, 'value')
        self.compute_btn.on_click(self.compute_btn_clicked)
        self.savepipe_btn.on_click(self.save_pipeline)
        self.filefolder_confirm_btn.on_click(self.on_filefolder_confirm_btn_click)
        self.file_view.on_click(self.on_file_buttons_click)
        self.file_download.on_click(self.on_file_buttons_click)
        self.file_upload.on_click(self.file_upload_click)
        self.file_delete.on_click(self.on_file_buttons_click)
        self.folder_select.param.watch(self.folder_select_changed, 'value')
        self.ontology_btn.on_click(self.ontology_btn_click)

        # Added event handler for recommendation button
        self.recommendation_btn.on_click(self.recommendation_btn_clicked)
        self.func_generator_btn.on_click(self.func_generator_btn_click)
        logger.info("Event handlers set up.")

    def create_layout(self):
        logger.info("Creating layout...")
        param_widget_input = pn.Column(
            #pn.layout.Divider(height=10, margin=(10)), 
            self.param_widget_tab,
            scroll=True,
            
        )
        widget_btns = pn.Row(self.savepipe_btn, self.pipeline_text, self.ontology_btn)
        widget_updownload = pn.Column(
            pn.Row(self.file_view, self.file_download),
            pn.Row(self.file_input, self.rules_button, width=280, height=50),
            pn.Row(self.file_upload, self.file_delete),
            scroll=True,
        )
        widget_files = pn.Column(
            self.folder_select,
            pn.Row(self.file_text, self.filefolder_confirm_btn),
            pn.layout.Divider(height=10, margin=(10)),
            self.file_table,
            widget_updownload,
            #width=300, 
            scroll=True,
        )
        widget_funcsel = pn.Column(
            self.funcfilecombo, 
            self.funccombo_pane,  # Use the placeholder here
            self.compute_btn, 
            widget_btns,
            #min_width=300
            scroll=True,
        )

        # Added recommendation widgets to the layout
        widget_recom = pn.Column(self.input_text,
                                 self.recomAPI_text,
                                 pn.Row(self.recommendation_btn, self.func_generator_btn),
                                scroll=True)
        
        self.app = pn.Row(
            pn.Column(widget_files,
                      min_width=200, 
                      max_width=300),
            pn.Column(widget_funcsel, 
                      self.output_text,
                      min_height=300,
                      min_width=300,
                      scroll=True),  

            pn.Column(
                pn.Column(param_widget_input,min_height=300,scroll=True),  
                pn.layout.Divider(height=10, margin=(10)), 
                pn.Column(widget_recom, scroll=True),  
                min_height=300,
                min_width=300,
                               
            ), 
 
            scroll=True,margin=(10, 0, 0, 0),
        )
        logger.info("Layout created.")

    def funcfilecombo_change(self, event):
        logger.info(f"funcfilecombo changed: {event.new}")
        selected_modules = event.new
        self.update_func_module(selected_modules)

    def update_func_module(self, module_names):
        logger.info(f"Updating functions for selected modules: {module_names}")
        if not module_names:
            self.funccombo_pane.objects = []
            self.output_text.value = "No modules selected."
            return

        self.modules = {}
        self.funcs = {}

        # Import selected modules dynamically
        for module_name in module_names:
            module_path = os.path.join(self.functionspace, module_name)
            if not os.path.exists(module_path):
                logger.warning(f"Module file {module_path} does not exist.")
                continue

            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(module_name[:-3], module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.modules[module_name] = module

            # Get functions from the module
            module_functions = self.get_functions_from_module(module, module_name)
            self.funcs.update(module_functions)

        # Update function combo box
        self.funccombo = self.create_multi_select_combobox(self.funcs)
        self.funccombo_pane.objects = [self.funccombo]
        # Set up event handler for the new funccombo
        self.funccombo.param.watch(self.funccombo_change, 'value')
        logger.info("Function combobox updated based on the selected modules.")

    def setup_full_func_mapping(self):
        if os.path.exists(self.full_func_mapping_path):
            os.remove(self.full_func_mapping_path)

        # Create a full function mapping file
        # key is the filename and value is a list of functions (name) in the file
        full_func_mapping = {}
        for py_file in self.py_files:
            file_path = os.path.join(self.functionspace, py_file)
            function_names = self.get_all_func_names(file_path)
            full_func_mapping[py_file] = function_names

        with open(self.full_func_mapping_path, 'w') as file:
            json.dump(full_func_mapping, file, indent=4)


    def get_all_func_names(self, filepath):
        with open(filepath, 'r') as file:
            content = file.read()
        
        # Parse the file content into an AST
        tree = ast.parse(content)
        
        # Extract all function definitions (ast.FunctionDef) from the tree
        function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
        return function_names


    def get_functions_from_module(self, module, module_name):
        functions = {}
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__ and not name.startswith('_'):
                func_key = f"{module_name[:-3]}.{name}"
                functions[func_key] = obj
        return functions

    def funccombo_change(self, event):
        logger.info(f"funccombo changed: {event.new}")
        self.output_text.value = ''
        list_funcs = self.funccombo.value
        logger.info(f"Selected functions: {list_funcs}")
        list_params = []
        for func_key in list_funcs:
            try:
                func = self.funcs[func_key]
                params = self.extract_parameter(func)
                list_params.append(params)
            except Exception as e:
                logger.error(f"Error extracting parameters for function {func_key}: {e}")
                list_params.append({})
        funcs_params = dict(zip(list_funcs, list_params))
        formatted_data = json.dumps(funcs_params, indent=4)
        self.json_editor.value = funcs_params
        self.json_text.value = formatted_data
        self.output_text.value = self.get_doc_string(formatted_data)
        
        #self.json_editor.expand_all()
    def json_text_change(self, event):
        """
        Handles changes to the JSON text input, preserving existing modules while adding
        new ones required by the JSON content.
        
        Parameters:
            event: The event object containing the new JSON text value
        """
        logger.info("JSON text input changed.")
        
        # Clean up the input text
        text = self.json_text.value
        text = re.sub(r'\bfalse\b', 'False', text, flags=re.IGNORECASE)
        text = text.replace("'", '"')
        
        try:
            # Parse the JSON text
            pipeline_dict = json.loads(text)
            pipeline_dict_json = json.dumps(pipeline_dict, indent=4)
            
            # Extract module names from function keys
            required_modules = set()
            for func_key in pipeline_dict.keys():
                if '.' in func_key:
                    module_name = func_key.split('.')[0] + '.py'
                    required_modules.add(module_name)
            
            # Get current modules and add new required ones
            current_modules = set(self.funcfilecombo.value)
            modules_to_add = required_modules - current_modules
            
            if modules_to_add:
                # Only add new modules, preserve existing ones
                updated_modules = list(current_modules | modules_to_add)
                logger.info(f"Adding new modules: {modules_to_add}")
                self.funcfilecombo.value = updated_modules
                
                # Update function modules with combined set
                self.update_func_module(updated_modules)
            
            # Update the JSON editor and text values
            self.json_text.value = pipeline_dict_json
            self.json_editor.value = pipeline_dict
            
            # Update function selection in funccombo if it exists
            if hasattr(self, 'funccombo'):
                func_keys = list(pipeline_dict.keys())
                current_funcs = set(self.funccombo.value)
                # Preserve existing function selections and add new ones
                updated_funcs = list(current_funcs | set(func_keys))
                self.funccombo.value = updated_funcs
            
            self.output_text.value = 'Input changed successfully! Existing modules preserved.'
            if modules_to_add:
                self.output_text.value += f'\nNew modules added: {", ".join(modules_to_add)}'
            
        except json.JSONDecodeError as e:
            error_msg = f'Error parsing JSON input: {str(e)}'
            self.output_text.value = error_msg
            logger.error(error_msg)
        except Exception as e:
            error_msg = f'Unexpected error processing input: {str(e)}'
            self.output_text.value = error_msg
            logger.error(error_msg)


    def json_toggle_clicked(self, event):
        logger.info(f"JSON toggle clicked: {event.new}")
        self.param_widget_tab.active = 1 if event.new else 0

    def json_editor_change(self, event):
        logger.info("JSON editor changed.")
        text = json.dumps(self.json_editor.value, indent=4)
        self.json_text.value = text

    def recommendation_btn_clicked(self, event):
        logger.info("Recommendation button clicked.")
        self.output_text.value = 'Asking AI for recommendation: \n'
        user_pipeline = self.json_editor.value
        user_query = self.input_text.value
        db_path = os.path.join(self.folder_path, 'KB.db')
        openai_api_key = self.recomAPI_text.value

        try:
            response_text = rc.pipeline_recommendation(db_path, user_query, user_pipeline, openai_api_key)
            self.output_text.value += response_text
            self.output_text.value += '\n\n=================================\n'
            response_text = rc.pipeline_parameters_recommendation(user_query, response_text, openai_api_key)

            text = str(response_text)
            text = re.sub(r"\b(false|False)\b", '"false"', text, flags=re.IGNORECASE)

            self.output_text.value += response_text

            services = json.loads(response_text)
            keys = list(services.keys())
            self.funccombo.value = keys

            rec_string = json.dumps(services, indent=4)
            self.json_editor.value = services
            logger.info("Recommendation process completed.")
        except Exception as e:
            self.output_text.value += f"\nError during recommendation: {e}"
            logger.error(f"Error during recommendation: {e}")


    def func_generator_btn_click(self, event):
        logger.info("Function generator button clicked.")
        query = self.input_text.value
        self.output_text.value = 'Generating function from query...'
        self.output_text.value = '\n\nBelow is the original query:\n\n' + query + '\n\n'
        generated_functions = generate_function(query)

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + 'auto_generated_function.py'

        temp_file_path = self.folder_path + '/temp/' + filename
        with open(temp_file_path, 'w') as f:
            f.write(generated_functions)

        flag, message, proposed_correction = function_validation_result(temp_file_path)

        folder = self.folder_path + '/functionspace'
        file_path = folder + '/' + filename

        self.output_text.value += '\n\n##################Generated & uploaded successfully!##############\n\n'
        self.output_text.value += f'The following function(s) have been generated and uploaded to SLEGO functionspace with filename: {filename}.\n'
        self.output_text.value += 'All functions are also validated and corrected if necessary.\n\n'
        self.output_text.value += message

        # add the generated functions' import statement to the file_path
        import_statements = extract_import_statements(temp_file_path)

        with open(file_path, 'w') as f:
            for statement in import_statements:
                f.write(statement)
                f.write('\n')
            f.write('\n')
            for _, correction in proposed_correction.items():
                f.write(correction)
                f.write('\n\n')

        if flag is False:
            
            self.output_text.value += "GENERATED FUNCTION WITH MODIFICATION\n\n"
            
        else:
            self.output_text.value += "NO MODIFICATION NEEDED\n\n"
            
        self.refresh_file_table()
        self.refresh_funcfilecombo()

        os.remove(temp_file_path)

    def compute_btn_clicked(self, event):
        logger.info("Compute button clicked.")
        self.input_text.value = 'Computing...'
        pipeline_dict = self.json_editor.value
        self.output_text.value = ''
        logger.info(f"Pipeline dict: {pipeline_dict}")

        for func_key, parameters in pipeline_dict.items():
            logger.info(f"Computing {func_key} with parameters {parameters}")
            self.input_text.value = f'Computing {func_key}...'
            try:
                start_time = time.time()
                function = self.funcs[func_key]
                result = function(**parameters)
                result_string = self.output_formatting(result)
                compute_time = time.time() - start_time

                self.output_text.value += f"\n===== {func_key} =====\n\n"
                self.output_text.value += f"Function computation time: {compute_time:.4f} seconds\n\n"
                self.output_text.value += result_string
                logger.info(f"Function {func_key} computed successfully.")
            except Exception as e:
                self.output_text.value += f"\n===== {func_key} =====\n\n"
                self.output_text.value += f"Error occurred: {str(e)}\n"
                logger.error(f"Error computing {func_key}: {e}")
            self.refresh_file_table()

        self.save_record('recordspace', pipeline_dict)
        self.input_text.value = 'Done!'
        self.on_filefolder_confirm_btn_click(None)
        self.refresh_file_table()


    def output_formatting(self, result):
        max_rows = 10
        max_words = 500
        max_chars = 2000

        final = ""

        # Check if the result is a pandas DataFrame
        if isinstance(result, pd.DataFrame):
            limited_df = result.head(max_rows)  # Limit to the first max_rows rows
            df_string = limited_df.to_string(index=False)
            final = df_string[:max_chars]  # Limit to max_chars characters
            if len(df_string) > max_chars:
                final += "\n... (truncated)"
        
        # Handle lists or dictionaries
        elif isinstance(result, (list, dict)):
            result_string = str(result)
            final = result_string[:max_chars]  # Limit to max_chars characters
            if len(result_string) > max_chars:
                final += "\n... (truncated)"

        else:
            result_string = str(result)
            words_iterator = iter(result_string.split())
            first_x_words = itertools.islice(words_iterator, max_words)
            truncated_result = " ".join(first_x_words)
            final = truncated_result[:max_chars]  # Limit to max_chars characters
            if len(truncated_result) > max_chars:
                final += "\n... (truncated)"

        return final

    

    def save_pipeline(self, event):
        logger.info("Save pipeline button clicked.")
        pipeline_name = self.pipeline_text.value if self.pipeline_text.value else '__'
        # Replace JavaScript-style JSON values with Python-compatible values
        text = self.json_text.value
        text = re.sub(r'\bfalse\b', '0', text, flags=re.IGNORECASE)
        text = re.sub(r'\btrue\b', '1', text, flags=re.IGNORECASE)
        text = re.sub(r'\bnull\b', 'None', text, flags=re.IGNORECASE)
        
        # Debugging: Print the processed text to verify JSON format
        logger.debug(f"Processed JSON text: {text}")
        
        try:
            # Parse the updated text as JSON
            data = json.loads(text)
            
            # Save the record using the pipeline_name and data
            self.save_record('knowledgespace', data, pipeline_name)
            self.on_filefolder_confirm_btn_click(None)
        except Exception as e:
            logger.error(f"Error saving pipeline: {e}")
            self.output_text.value += f"\nError saving pipeline: {e}"



    def open_with_default_app(self, file_path):
        """
        Opens a file with the system's default application based on the operating system.
        
        Parameters:
        - file_path (str): The path to the file to open.
        """
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.call(["open", file_path])
            elif platform.system() == "Windows":  # Windows
                os.startfile(file_path)
            elif platform.system() == "Linux":  # Linux
                subprocess.call(["xdg-open", file_path])
            else:
                print("Unsupported operating system.")
            print(f"The file at {file_path} was opened with its default application.")
        except Exception as e:
            print(f"An error occurred while trying to open the file: {e}")


    def create_download_script(self, file_path):
        """
        Creates a JavaScript function to trigger download programmatically.
        """
        return f"""
            function downloadFile() {{
                const link = document.createElement('a');
                link.href = '{file_path}';
                link.download = '{os.path.basename(file_path)}';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }}
            downloadFile();
        """

    def download_link(self, file_path):
        """
        Generates and executes a download link for a file in Codespace.
        """
        try:
            # Create and execute JavaScript to trigger download
            script = self.create_download_script(file_path)
            display(Javascript(script))
            return f"Downloading {os.path.basename(file_path)}..."
        except Exception as e:
            print(f"Error generating download link: {e}")


    def file_upload_click(self, event):
        if self.file_input.filename:
            filename = self.file_input.filename
            self.output_text.value = f'Uploading {filename}...'

            file_content = self.file_input.value

            temp_file_path = self.folder_path + '/temp/' + filename
            with open(temp_file_path, 'wb') as f:
                f.write(file_content)

            flag, message, proposed_correction = function_validation_result(temp_file_path)
            
            folder = self.folder_path + '/' + self.file_text.value
            file_path = folder + '/' + filename

            # If a file already exists with the same name, add a timestamp to the filename
            if os.path.exists(file_path):
                filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + filename
                file_path = folder + '/' + filename
                self.output_text.value += f'File with the same name already exists. Renaming to {filename}...'

            # add the generated functions' import statement to the file_path
            import_statements = extract_import_statements(temp_file_path)
            
            with open(file_path, 'w') as f:
                for statement in import_statements:
                    f.write(statement)
                    f.write('\n')
                f.write('\n')
                for _, correction in proposed_correction.items():
                    f.write(correction)
                    f.write('\n\n')

            if flag is False:
                self.output_text.value = "\n\n##################File Upload Success BUT with modification##################\n\n" + message
                                
            else:

                self.output_text.value += f'\n\n##################{filename} uploaded successfully!##############\n\n' + message
                
            self.refresh_file_table()
            self.refresh_funcfilecombo()

            os.remove(temp_file_path)
        else:
            self.output_text.value = 'Please select a file to upload!'


    def on_file_buttons_click(self, event):
        logger.info(f"File button '{event.obj.name}' clicked.")
        self.output_text.value = ''
        file_list = self.file_table.selected_dataframe['Filter Files :'].tolist()
        if file_list:
            for filename in file_list:
                file_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'), filename)
                if event.obj.name == 'View':
                    self.open_with_default_app(file_path)
                    with open(file_path, 'r') as file:
                        content = file.read()
                    self.output_text.value += f"\n===== {filename} =====\n{content}\n"
                elif event.obj.name == 'Download':
                    self.output_text.value = 'Initiating download...\n'
                    result = self.download_link(file_path)
                elif event.obj.name == 'Upload' and not self.file_input.filename:
                    self.output_text.value = 'Please use the file input widget to upload!'
                elif event.obj.name == 'Delete':
                    # Confirm deletion
                    self.output_text.value = 'Are you sure you want to delete the selected file(s)? Please type "delete" in the user input textbox.'
                    #user input type are delete:
                    if self.input_text.value == 'delete':
                        os.remove(file_path)
                        self.output_text.value += f"\nDeleted file: {filename}"
                        logger.info(f"Deleted file: {filename}")
                        self.refresh_file_table()  # Refresh table after deletion
                    else:
                        #self.output_text.value += f"\nDeletion cancelled."
                        logger.info(f"Deletion cancelled.")
    
        else:
            self.output_text.value = 'Please select a file to perform the action.'

    def on_filefolder_confirm_btn_click(self, event):
        logger.info("File folder confirm button clicked.")
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        if os.path.exists(selected_folder_path):
            file_list = os.listdir(selected_folder_path)
            df_file = pd.DataFrame(file_list, columns=['Filter Files :'])
            self.file_table.value = df_file
            logger.info(f"Updated file table with files from {selected_folder_path}")
        else:
            logger.warning(f"Folder {selected_folder_path} does not exist.")
            self.file_table.value = pd.DataFrame()

    def folder_select_changed(self, event):
        logger.info(f"Folder selected: {event.new}")
        self.file_text.value = '/' + str(self.folder_select.value)
        self.on_filefolder_confirm_btn_click(None)

    def get_doc_string(self, pipeline_text):
        output = ''
        data = json.loads(pipeline_text)
        for func_key in data.keys():
            output += f"===== {func_key} =====\n"
            try:
                func = self.funcs[func_key]
                doc = func.__doc__
                output += doc if doc else 'No docstring available.\n'
            except AttributeError:
                output += 'Function not found.\n'
        return output

    def save_record(self, space_key, data, filename=None):
        space = self.config.get(space_key)
        if not space:
            logger.error(f"Invalid space key: {space_key}")
            return

        if filename is None:
            filename = datetime.now().strftime("record_%Y%m%d_%H%M%S.json")
        else:
            filename = filename + '.json'

        full_path = os.path.join(space, filename)
        try:
            with open(full_path, "w") as file:
                json.dump(data, file, indent=4)
            logger.info(f"Record saved to {full_path}")
            self.refresh_file_table()
        except Exception as e:
            logger.error(f"Error saving record: {e}")
            self.output_text.value += f"\nError saving record: {e}"

    def refresh_file_table(self):
        logger.info("Refreshing file table...")
        selected_folder_path = os.path.join(self.folder_path, self.file_text.value.lstrip('/'))
        if os.path.exists(selected_folder_path):
            file_list = os.listdir(selected_folder_path)
            df_file = pd.DataFrame(file_list, columns=['Filter Files :'])
            self.file_table.value = df_file
        else:
            self.file_table.value = pd.DataFrame()

    def refresh_funcfilecombo(self):
        # Include all .py files in the functionspace directory
        self.py_files = [f for f in os.listdir(self.functionspace) if f.endswith('.py')]
        self.funcfilecombo.options = self.py_files
        logger.info(f"Refreshed funcfilecombo with {len(self.py_files)} files.")

    def create_multi_select_combobox(self, funcs):
        options = list(funcs.keys())
        multi_combobox = pn.widgets.MultiChoice(name='Select Components:', options=options, height=150)
        return multi_combobox

    def extract_parameter(self, func):
        signature = inspect.signature(func)
        parameters = signature.parameters
        parameter_dict = {}
        for name, param in parameters.items():
            if param.default != inspect.Parameter.empty:
                parameter_dict[name] = param.default
            else:
                parameter_dict[name] = None
        return parameter_dict



    # Step 2: Define the construct_subgraph function
    def construct_subgraph(self, file_path, query, output_file):
        """
        Executes a CONSTRUCT query on an RDF graph and saves the resulting subgraph to a file.

        Args:
            file_path (str): Path to the input RDF Turtle file.
            query (str): SPARQL CONSTRUCT query string.
            output_file (str): Path to the output Turtle file.
        """
            # Step 1: Define the style

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
    def graph_to_results(self, graph):
        results = []
        for subj, pred, obj in graph:
            row = [('subject', subj), ('predicate', pred), ('object', obj)]
            results.append(row)
        return results

    # Step 4: Helper Function to Extract Local Names
    def get_local_name(self, uri):
        if isinstance(uri, rdflib.term.URIRef):
            uri = str(uri)
        if '#' in uri:
            return uri.split('#')[-1]
        elif '/' in uri:
            return uri.rstrip('/').split('/')[-1]
        else:
            return uri

    # Step 5: Visualization Function
    def visualize_query_results_interactive(self, results):
        style = {
            "color": "gray",
            "shape": "ellipse",
            "size": 10
        }

        G = nx.DiGraph()
        node_types = {}

        # Build the graph
        for row in results:
            row_dict = dict(row)
            subj = self.get_local_name(row_dict['subject'])
            pred = row_dict['predicate']
            obj = self.get_local_name(row_dict['object'])
            
            # Capture rdf:type relationships to identify node types
            if str(pred) == rdflib.RDF.type:
                node_types[subj] = obj

            G.add_edge(subj, obj, label= self.get_local_name(pred))

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
            node_type = self.get_local_name(node_type_uri) if node_type_uri else 'Unknown'
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

    def extract_and_visualize_subgraph(self, file_path, subgraph_file, query):
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
        results = self.graph_to_results(g)

        # Visualize the subgraph
        self.visualize_query_results_interactive(results)

    def ontology_btn_click(self, event):

        sparql_query = self.input_text.value.strip()  # Ensure no leading/trailing spaces
        if not sparql_query:
            self.output_text.value = "Error: No SPARQL query provided. Please input a valid query."
            logger.error("No SPARQL query provided.")
            return
        
        # Example usage of the function
        file_path = 'slegospace/ontologyspace/hfd.ttl'
        subgraph_file = 'slegospace/ontologyspace/hfd_subgraph.ttl'

        # Call the function with the query as a parameter
        self.extract_and_visualize_subgraph(file_path, subgraph_file, sparql_query)


    def create_validation_rules_message(self):
        message = "Microservice Rules:\n\n"
        error_rule_message = ""
        warning_rule_message = ""
        for rule in validation_rules:
            if rule.issue_type == 'ERROR':
                error_rule_message += f"{rule.description}\n"
            if rule.issue_type == 'WARNING':
                warning_rule_message += f"{rule.description}\n"
        message = f"**The following rules will lead to unsuccessful upload if not followed**\n\n{error_rule_message}\n\n"
        message += f"**The following rules are recommended to be followed**\n\n{warning_rule_message}"
        return message


    def create_template(self):
        template = pn.template.MaterialTemplate(
            title='SLEGO - Software Lego: A Collaborative and Modular Architecture for Data Analytics'
        )

        template.modal.append(self.create_validation_rules_message())

        self.template = template

    def toggle_rule(self, event):
        self.template.open_modal()


    def run(self, modules=None):
        """
        Run the app, optionally with specific modules.
        Args:
            modules (list): List of modules to load.
        """


        # Software Introduction section
        software_intro = pn.pane.Markdown("""
        # Software Introduction

        **SLEGO (Software-Lego)** is a collaborative analytics platform designed to bridge the gap between experienced developers and novice users. It leverages a cloud-based environment with modular, reusable microservices, enabling developers to share their analytical tools and workflows. Novice users can construct comprehensive analytics pipelines through an intuitive graphical user interface (GUI) without requiring programming skills. Supported by a knowledge base and a Large Language Model (LLM) powered recommendation system, SLEGO enhances the selection and integration of microservices, increasing the efficiency of analytics pipeline construction.

        **Key Features:**
        - **Modular Microservices:** Share and integrate analytical tools and workflows seamlessly.
        - **User-Friendly GUI:** Build analytics pipelines through a simple drag-and-drop interface.
        - **LLM-Powered Recommendations:** Enhance microservice selection and integration with AI-driven guidance.
        - **Collaborative Environment:** Promote resource reusability and team collaboration.

        For more detailed information, refer to the [SLEGO paper](https://arxiv.org/abs/2406.11232).
        """, min_width=600)

        # About the Creator section
        creator_info = pn.pane.Markdown("""
        # About the Creator

        **Siu Lung Ng** is a PhD student at the School of Computer Science and Engineering, University of New South Wales (UNSW), Sydney, Australia. Under the supervision of Professor Fethi Rabhi, his research focuses on developing collaborative analytics platforms that democratize data analytics by integrating modular design, knowledge bases, and recommendation systems, fostering a more inclusive and efficient analytical environment.
        """, min_width=600)


        # Main application content (placeholder)
        main_content = pn.pane.Markdown("""
        # Main Application

        This is the main content area of the application.
        """)

        
        main_tabs = pn.Tabs(('SLEGO-App',self.app),
            ('Software Introduction', software_intro),
            ('About the Creator', creator_info),
            scroll=True, )

        logger.info("Running the app...")
        if modules:
            logger.info(f"Loading specified modules: {modules}")
            #add the module names to the function space
            self.funcfilecombo.value = modules   
            #self.update_func_module(modules)


        if not self.is_colab_runtime():
            self.rules_button.on_click(self.toggle_rule)
            self.template.main.append(main_tabs)
            # template.sidebar,
            # template.collapsed_sidebar = True
            
            self.template.show()
            self.template.servable()
            logger.info("App is running in non-Colab environment.")
        else:
            from IPython.display import display
            display(self.app)
            logger.info("App is running in Colab environment.")


    @staticmethod
    def is_colab_runtime():
        return 'google.colab' in sys.modules

