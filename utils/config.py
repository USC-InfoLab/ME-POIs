import argparse
import json
from typing import Optional

def args_parser(config_file: Optional[str] = None) -> argparse.Namespace:
    """
    Parses command-line arguments and optionally loads parameters from a JSON config file.
    """
    parser = argparse.ArgumentParser()

    if config_file:
        try:
            with open(config_file, "r") as file:
                arg_vals = json.load(file)
            
            # Add arguments from the JSON config file
            for key, value in arg_vals.items():
                parser.add_argument(
                    f"--{key}", type=type(value), default=value, help=f"Description for {key}"
                )
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_file} not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Error parsing JSON from {config_file}.")

    args, _ = parser.parse_known_args()
    return args

