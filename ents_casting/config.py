"""Central configuration loader."""

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()

# Add parameters that should only be changed by experienced users:
config["timezone"] = "UTC+1"
config["path_data_meteo"] = "ents_casting/data/meteo"
config["path_data_measured"] = "ents_casting/data/measured_data"
config["path_data_forecast_model_params"] = "ents_casting/data/forecast_models"
config["path_data_ar_model_params"] = "ents_casting/data/ar_model_params"
config["path_data_long_term_scenarios"] = "ents_casting/data/long_term_scenarios"
config["path_data_forecasts"] = "ents_casting/data/forecasts"

# Create data directories if they do not exist
cwd = Path.cwd()
for key, path in config.items():
    if key.startswith("path_"):
        dir_path = cwd / path
        dir_path.mkdir(parents=True, exist_ok=True)
