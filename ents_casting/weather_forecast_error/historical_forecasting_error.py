"""
This module calculates weather forecast errors of weather parameters based on the output of
meteo_downloaders/previous_weather_forecasts.py and saves the results to CSV files.
"""

import math
from datetime import timedelta
from pathlib import Path
import logging
import pandas as pd

from ents_casting.config import config


def main() -> None:
    logging.info("Calculating historical forecast errors.")

    # Main location
    cwd = Path.cwd()
    file_path = cwd / config["path_data_meteo"] / "forecasts_main.csv"
    forecasts = pd.read_csv(file_path, index_col="time", parse_dates=["time"])
    forecast_error_dfs = extract_errors(forecasts)
    # Save to CSV files
    for param, df in forecast_error_dfs.items():
        output_file = cwd / config["path_data_meteo"] / f"historical_forecast_error_{param}.csv"
        df.to_csv(output_file)
        print(f"Saved historical forecast error for '{param}' to {output_file}.")

    # Onshore locations
    for i, _ in enumerate(config["onshore_locations"]):
        file_path = cwd / config["path_data_meteo"] / f"forecasts_onshore_{i}.csv"
        forecasts = pd.read_csv(file_path, index_col="time", parse_dates=["time"])
        forecast_error_dfs = extract_errors(forecasts)
        for param, df in forecast_error_dfs.items():
            output_file = (
                cwd
                / config["path_data_meteo"]
                / f"historical_forecast_error_onshore_{i}_{param}.csv"
            )
            df.to_csv(output_file)
            print(
                f"Saved historical forecast error for onshore location {i}, '{param}' to {output_file}."
            )

    # Offshore locations
    for i, _ in enumerate(config["offshore_locations"]):
        file_path = cwd / config["path_data_meteo"] / f"forecasts_offshore_{i}.csv"
        forecasts = pd.read_csv(file_path, index_col="time", parse_dates=["time"])
        forecast_error_dfs = extract_errors(forecasts)
        for param, df in forecast_error_dfs.items():
            output_file = (
                cwd
                / config["path_data_meteo"]
                / f"historical_forecast_error_offshore_{i}_{param}.csv"
            )
            df.to_csv(output_file)
            print(
                f"Saved historical forecast error for offshore location {i}, '{param}' to {output_file}."
            )


def extract_errors(forecasts: pd.DataFrame) -> pd.DataFrame:
    main_params = [col for col in forecasts.columns if "previous_day" not in col]

    # Convert columns to dicts for fast lookup
    forecast_data = {col: forecasts[col].to_dict() for col in forecasts.columns}

    # Initialize error containers
    forecast_times = forecasts.index[:-144][::24]
    forecast_error_data = {
        param: {forecast_time: {0: 0.0} for forecast_time in forecast_times}
        for param in main_params
    }

    # Calculate forecast errors
    for forecast_time in forecast_times:
        for lead_time in range(1, 144):
            # forecast_time: time at which forecast is for
            # lead_time: forecast_time - time at which forecast is made
            target_time = forecast_time + timedelta(hours=lead_time)
            lead_day = math.ceil(lead_time / 24)

            for param in main_params:
                real_value = forecast_data[param].get(target_time)
                forecast_col = f"{param}_previous_day{lead_day}"
                forecast_value = forecast_data.get(forecast_col, {}).get(target_time)

                if real_value is None or forecast_value is None:
                    raise ValueError(f"Missing data for parameter '{param}' at time {target_time}.")
                forecast_error_data[param][forecast_time][lead_time] = forecast_value - real_value

    # Convert to DataFrames
    forecast_error_dfs = {
        param: pd.DataFrame.from_dict(error_dict, orient="index").rename_axis("time")
        for param, error_dict in forecast_error_data.items()
    }

    return forecast_error_dfs


if __name__ == "__main__":
    main()
