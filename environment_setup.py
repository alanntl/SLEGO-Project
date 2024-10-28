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
    
    # Set default local_repo_path if not provided
    if local_repo_path is None:
        local_repo_path = os.getcwd()

    # Ensure we're using absolute paths
    local_repo_path = os.path.abspath(local_repo_path)

    if env == 'colab':
        print("Setting up configuration for Google Colab environment.")
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        drive_root = '/content/drive/MyDrive'
        config['drive_folder'] = drive_root
        config['drive_mainfolder'] = os.path.join(drive_root, 'SLEGO')
    else:
        print("Setting up configuration for Local or GitHub Codespaces environment.")
        config['drive_folder'] = local_repo_path
        config['drive_mainfolder'] = local_repo_path

    # Include use_local_repo and local_repo_path in the config
    config['use_local_repo'] = use_local_repo
    config['local_repo_path'] = local_repo_path

    # Define base paths ensuring no nested slegospace
    base_path = config['drive_mainfolder']
    if 'slegospace' in base_path:
        # If we're already in a slegospace directory, go up to parent
        base_path = os.path.dirname(base_path)
        while os.path.basename(base_path) == 'slegospace':
            base_path = os.path.dirname(base_path)
    
    # Set up workspace paths
    config['folder_path'] = os.path.join(base_path, 'slegospace')
    config['dataspace'] = os.path.join(config['folder_path'], 'dataspace')
    config['recordspace'] = os.path.join(config['folder_path'], 'recordspace')
    config['functionspace'] = os.path.join(config['folder_path'], 'functionspace')
    config['knowledgespace'] = os.path.join(config['folder_path'], 'knowledgespace')
    config['ontologyspace'] = os.path.join(config['folder_path'], 'ontologyspace')
    
    # Update environment variables with correct paths
    os.environ['DRIVE_MAINFOLDER'] = config['drive_mainfolder']
    os.environ['DRIVE_FOLDER'] = config['drive_folder']

    print("\nConfiguration settings:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return config

def is_valid_workspace(path: str) -> bool:
    """Check if the given path is a valid workspace with all required folders."""
    required_spaces = ['dataspace', 'recordspace', 'functionspace', 'knowledgespace', 'ontologyspace']
    return all(os.path.exists(os.path.join(path, space)) for space in required_spaces)

def setup_workspace(config: Dict[str, Any]):
    """Set up the workspace folders and change the working directory."""
    print("\nSetting up workspace folders...")
    
    # Check if current directory is already a valid workspace
    current_dir = os.path.abspath(os.getcwd())
    if is_valid_workspace(current_dir):
        print("Already in a valid workspace. No changes needed.")
        return

    # Create workspace folders if they don't exist
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

    # Change directory to workspace root
    desired_dir = os.path.abspath(config['folder_path'])
    if desired_dir != current_dir:
        try:
            os.chdir(desired_dir)
            print(f"Working directory changed to: {os.getcwd()}")
        except PermissionError as e:
            print(f"PermissionError when changing directory: {e}")
    else:
        print(f"Already in the workspace directory: {desired_dir}")

    if detect_environment() == 'colab':
        from google.colab import files
        print(f"Viewing folder in Colab: {config['folder_path']}")
        files.view(config['folder_path'])

def setup_environment(use_local_repo: bool = True, local_repo_path: Optional[str] = None):
    """Set up the environment and workspace."""
    print("Starting environment setup...")
    config = get_environment_config(use_local_repo=use_local_repo, local_repo_path=local_repo_path)
    setup_workspace(config)
    print("Environment setup completed.")
    return config