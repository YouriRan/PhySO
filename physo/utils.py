import json


def load_config(run_config):
    if isinstance(run_config, dict):
        return run_config
    elif isinstance(run_config, str):
        with open(run_config, "r") as f:
            return json.load(f)
    else:
        raise ValueError("config should be filename or dict")
