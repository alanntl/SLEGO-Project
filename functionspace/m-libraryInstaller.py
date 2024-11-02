import subprocess
import sys
import importlib

def install_and_import(package_name):
    """
    Installs and imports a Python package.
    
    Parameters:
    package_name (str): The name of the package to install and import.
    
    Returns:
    module: The imported module.
    
    Raises:
        subprocess.CalledProcessError: If pip installation fails
        ImportError: If module cannot be imported after installation
    """
    try:
        # Try to import the package
        return importlib.import_module(package_name)
    except ImportError:
        # If the package is not installed, install it
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            # Import the package after installation
            return importlib.import_module(package_name)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to install package {package_name}: {str(e)}")