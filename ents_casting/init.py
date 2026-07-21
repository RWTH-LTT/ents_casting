# Create data directories if they do not exist
from pathlib import Path

paths = [
    "ents_casting/data/measured_data",
    "ents_casting/data/forecast_models",
    "ents_casting/data/ar_model_params",
    "ents_casting/data/long_term_scenarios",
    "ents_casting/data/forecasts",
]

cwd = Path.cwd()
for path in paths:
    dir_path = cwd / path
    dir_path.mkdir(parents=True, exist_ok=True)
