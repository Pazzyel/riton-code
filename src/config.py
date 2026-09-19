import os
from pathlib import Path
from typing import Literal

import yaml

with Path("config.yaml").open("r", encoding="utf-8") as config_stream:
    config_file: dict = yaml.safe_load(config_stream)
model_config: dict = config_file["model"]

MODEL_ID: str = model_config["name"]
BASE_URL: str = model_config["base_url"]
API_KEY: str = os.getenv(model_config["api_key_path_var"],"")
MAX_TOKENS: int = 8192

permission_config: dict = config_file["permission"]
PERMISSION_MODE: Literal["default", "plan", "auto", "allow"] = permission_config["mode"]
sandbox_config: dict = permission_config.get("sandbox", {})
SANDBOX_SETTING: Literal["read", "workspace", "all"] = sandbox_config.get("setting", "workspace")
SANDBOX_NETWORK_ACCESS: bool = sandbox_config.get("network-access", False)

if PERMISSION_MODE not in {"default", "plan", "auto", "allow"}:
    raise ValueError(f"Unsupported permission mode: {PERMISSION_MODE}")
if SANDBOX_SETTING not in {"read", "workspace", "all"}:
    raise ValueError(f"Unsupported sandbox setting: {SANDBOX_SETTING}")
if not isinstance(SANDBOX_NETWORK_ACCESS, bool):
    raise ValueError("permission.sandbox.network-access must be a boolean")

if not API_KEY:
    raise ValueError(f"API key not found. Please set the environment variable {model_config['api_key_path_var']} with your API key.")
