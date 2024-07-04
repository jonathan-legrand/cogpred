from pyaml_env import parse_config

def get_config(fpath="config.yml"):
    config = parse_config(path=fpath)
    return config

