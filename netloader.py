'''netloader.py

MKP - August  2025
'''
import os
import yaml

def get_params(filename, config_folder='network_params'):
    """
    Load neural network hyperparameters from a YAML file.
    
    Args:
        filename (str): Name of the YAML file (with or without .yaml/.yml extension).
        config_folder (str): Subfolder where config files are stored.

    Returns:
        dict: Dictionary of parameters.
    """
    # Ensure .yaml or .yml extension
    if not (filename.endswith('.yaml') or filename.endswith('.yml')):
        filename += '.yaml'

    filepath = os.path.join(config_folder, filename)

    with open(filepath, "r") as f:
        params = yaml.safe_load(f)

    return params

