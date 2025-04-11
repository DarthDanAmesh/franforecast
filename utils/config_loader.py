# Loads configuration settings (e.g., minimum rows, default parameters)
import json
import os

def load_config(config_file="config.json"):
    """
    Load configuration settings from a JSON file.
    These settings can include default parameters, minimum rows, and other global configurations.
    Parameters
    ----------
    config_file : str
        Path to the configuration file.
    Returns
    -------
    dict
        Configuration dictionary.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    with open(config_file, "r") as f:
        config = json.load(f)
    return config