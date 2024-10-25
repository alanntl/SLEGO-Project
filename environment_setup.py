import os
import sys
import subprocess
import platform
from typing import Dict, Any, Optional
import os
import sys
import platform
from typing import Dict, Any, Optional

def detect_environment() -> str:
    """Detect the current runtime environment."""
    print("Detecting environment...")
    if 'google.colab' in sys.modules:
        print("Environment detected: Google Colab")
        return 'colab'
    elif os.environ.get('CODESPACES', 'false').lower() == 'true':
        print("Environment detected: GitHub Codespaces")
        return 'github-codespaces'
    else:
        print("Environment detected: Local Jupyter")
        return 'local-jupyter'

def get_environment_config(use_local_repo: bool = True, local_repo_path: Optional[str] = None) -> Dict[str, Any]:
    """Get the configuration based on the current runtime environment."""
    print("Getting environment configuration...")
    config = {}
    env = detect_environment()

    if env == 'colab':
        print("Setting up configuration for Google Colab environment.")
        # Mount Google Drive
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        drive_root = '/content/drive/MyDrive'
        config['drive_folder'] = drive_root
        config['drive_mainfolder'] = os.path.join(drive_root, 'SLEGO')
    elif env == 'github-codespaces':
        print("Setting up configuration for GitHub Codespaces environment.")
        home_dir = os.path.expanduser('~')
        config['drive_folder'] = home_dir
        config['drive_mainfolder'] = os.path.join(home_dir, 'SLEGO')
    else:  # local-jupyter or other environments
        print("Setting up configuration for Local Jupyter environment.")
        if use_local_repo:
            if local_repo_path:
                config['drive_mainfolder'] = local_repo_path
                config['drive_folder'] = os.path.dirname(local_repo_path)
            else:
                raise ValueError("local_repo_path must be provided when use_local_repo is True")
        else:
            # If not using local repo, set default drive_mainfolder
            home_dir = os.path.expanduser('~')
            config['drive_folder'] = home_dir
            config['drive_mainfolder'] = os.path.join(home_dir, 'SLEGO')

    # Include use_local_repo and local_repo_path in the config
    config['use_local_repo'] = use_local_repo
    config['local_repo_path'] = local_repo_path

    config['slego_env'] = os.path.join(config['drive_folder'], 'slego_env_v0_0_1')
    config['requirements_file'] = os.path.join(config['drive_mainfolder'], 'requirements.txt')

    # Set up workspace folders
    config['folder_path'] = os.path.join(config['drive_mainfolder'], 'slegospace')
    config['dataspace'] = os.path.join(config['folder_path'], 'dataspace')
    config['recordspace'] = os.path.join(config['folder_path'], 'recordspace')
    config['functionspace'] = os.path.join(config['folder_path'], 'functionspace')
    config['knowledgespace'] = os.path.join(config['folder_path'], 'knowledgespace')
    config['ontologyspace'] = os.path.join(config['folder_path'], 'ontologyspace')

    os.environ['DRIVE_MAINFOLDER'] = config['drive_mainfolder']
    os.environ['DRIVE_FOLDER'] = config['drive_folder']

    print("Configuration settings:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return config

def setup_repository(config: Dict[str, Any]):
    """Assume repository is already set up; do nothing."""
    print("Using existing repository; no action needed in setup_repository.")

def setup_workspace(config: Dict[str, Any]):
    """Set up the workspace folders and change the working directory."""
    print("Setting up workspace folders...")
    workspace_folders = [
        config['folder_path'],
        config['dataspace'],
        config['recordspace'],
        config['functionspace'],
        config['knowledgespace'],
        config['ontologyspace'],
    ]
    for folder in workspace_folders:
        if not os.path.exists(folder):
            print(f"Creating folder: {folder}")
            os.makedirs(folder, exist_ok=True)
        else:
            print(f"Folder already exists: {folder}")

    # Only change directory if folder_path is different from current directory
    desired_dir = os.path.abspath(config['folder_path'])
    current_dir = os.path.abspath(os.getcwd())
    if desired_dir != current_dir:
        try:
            os.chdir(desired_dir)
            print(f"Working directory changed to: {os.getcwd()}")
        except PermissionError as e:
            print(f"PermissionError when changing directory: {e}")
    else:
        print(f"Already in the workspace directory: {config['folder_path']}")

    if detect_environment() == 'colab':
        from google.colab import files
        print(f"Viewing folder in Colab: {config['folder_path']}")
        files.view(config['folder_path'])

def setup_environment(use_local_repo: bool = True, local_repo_path: Optional[str] = None):
    """Set up the environment and workspace, assuming repository is already set up."""
    print("Starting environment setup...")
    config = get_environment_config(use_local_repo=use_local_repo, local_repo_path=local_repo_path)
    setup_repository(config)
    # Uncomment the next line if virtual environment setup is needed
    # setup_virtual_environment(config)
    setup_workspace(config)
    print("Environment setup completed.")
    return config
