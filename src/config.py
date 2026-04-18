import yaml
import os

config_file: dict = yaml.safe_load(open("config.yaml", "r"))
model_config: dict = config_file["model"]

MODEL_ID: str = model_config["name"]
BASE_URL: str = model_config["base_url"]
API_KEY: str = os.getenv(model_config["api_key_path_var"],"")
MAX_TOKENS: int = 8192

if not API_KEY:
    raise ValueError(f"API key not found. Please set the environment variable {model_config['api_key_path_var']} with your API key.")