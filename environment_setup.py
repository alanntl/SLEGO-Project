import os
import sys
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
REQUIRED_SPACES = ['dataspace', 'recordspace', 'functionspace', 'knowledgespace', 'ontologyspace', 'temp']

# Utility functions
def detect_environment() -> str:
    """Detect the current runtime environment."""
    if 'google.colab' in sys.modules:
        logging.info("Environment detected: Google Colab")
        return 'colab'
    elif os.environ.get('CODESPACES', 'false').lower() == 'true':
        logging.info("Environment detected: GitHub Codespaces")
        return 'github-codespaces'
    elif 'ipykernel' in sys.modules:
        logging.info("Environment detected: Jupyter Notebook or JupyterLab")
        return 'jupyter'
    else:
        logging.info("Environment detected: Standalone Python script")
        return 'local-script'

# @contextmanager
# def change_directory(destination: str):
#     """Temporarily change the working directory."""
#     current_dir = os.getcwd()
#     try:
#         os.chdir(destination)
#         yield
#     finally:
#         os.chdir(current_dir)


def is_valid_workspace(path: str) -> bool:
    """Check if the given path is a valid workspace with all required folders."""
    return all(os.path.exists(os.path.join(path, space)) for space in REQUIRED_SPACES)

def setup_workspace(config: Dict[str, Any]):
    """Set up the workspace folders and change the working directory."""
    current_dir = os.path.abspath(os.getcwd())
    if is_valid_workspace(current_dir):
        logging.info("Already in a valid workspace. No changes needed.")
        return

    # Create workspace folders if they don't exist
    for folder in [config['folder_path']] + [config[space] for space in REQUIRED_SPACES]:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            logging.info(f"Creating folder: {folder}")

    # # Change directory to workspace root
    # desired_dir = os.path.abspath(config['folder_path'])
    # if desired_dir != current_dir:
    #     with change_directory(desired_dir):
    #         logging.info(f"Working directory changed to: {os.getcwd()}")

    # Handle Colab-specific folder view
    if config.get('environment') == 'colab':
        try:
            from google.colab import files
            logging.info(f"Viewing folder in Colab: {config['folder_path']}")
            files.view(config['folder_path'])
        except ImportError:
            logging.warning("Not running in Colab. Skipping Colab-specific operations.")

def setup_environment(use_local_repo: bool = True, local_repo_path: Optional[str] = None):
    """Set up the environment and workspace."""
    logging.info("Starting environment setup...")
    config = get_environment_config(use_local_repo=use_local_repo, local_repo_path=local_repo_path)
    setup_workspace(config)
    logging.info("Environment setup completed.")
    return config


def get_environment_config(use_local_repo: bool = True, local_repo_path: Optional[str] = None) -> Dict[str, Any]:
    """Get the configuration based on the current runtime environment."""
    env = detect_environment()

    # Set default local_repo_path if not provided
    if local_repo_path is None:
        local_repo_path = os.getcwd()

    # Ensure we're using absolute paths
    local_repo_path = os.path.abspath(local_repo_path)
    config = {
        'use_local_repo': use_local_repo,
        'local_repo_path': local_repo_path,
    }

    if env == 'colab':
        from google.colab import drive
        logging.info("Setting up configuration for Google Colab environment.")
        drive.mount('/content/drive', force_remount=True)
        drive_root = '/content/drive/MyDrive'
        config['drive_folder'] = drive_root
        config['drive_mainfolder'] = os.path.join(drive_root, 'SLEGO')
    else:
        logging.info("Setting up configuration for Local or GitHub Codespaces or Jupyter environment.")
        config['drive_folder'] = local_repo_path
        config['drive_mainfolder'] = local_repo_path

    # # Define base paths ensuring no nested slegospace
    base_path = config['drive_mainfolder']
    # if 'slegospace' in base_path:
    #     base_path = os.path.dirname(base_path)
    #     while os.path.basename(base_path) == 'slegospace':
    #         base_path = os.path.dirname(base_path)

    # Update configuration with folder paths
    config['folder_path'] = os.path.join(base_path)
    config.update({space: os.path.join(config['folder_path'], space) for space in REQUIRED_SPACES})

    # Update environment variables with correct paths
    os.environ['DRIVE_MAINFOLDER'] = config['drive_mainfolder']
    os.environ['DRIVE_FOLDER'] = config['drive_folder']

    logging.info("Configuration settings: %s", config)
    return config
