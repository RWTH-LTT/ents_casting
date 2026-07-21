"""Module to fit AR(2) models for weather forecast errors."""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List

from ents_casting.ar_models.short_term_error import ShortTermArFitter
from ents_casting.config import config


def main() -> None:
    """
    Main function to fit AR(2) models for weather parameters in config["main_params"].
    """
    weather_params = get_all_weather_params()
    for param in weather_params:
        print(f"Fitting AR(2) model for {param} forecast errors.")
        hist_values, hist_forecasts, hist_forecast_errors = prepare_weather_forecast_errors(param)

        ar_model_fitter = ShortTermArFitter(param=param, n_bins=10)
        bin_edges, hist_errors_by_forecast_bin, hist_errors_by_true_value_bin, ar_params = (
            ar_model_fitter.fit_short_term_AR(hist_values, hist_forecasts, hist_forecast_errors)
        )
        ar_model_fitter.save_parameters(
            bin_edges, hist_errors_by_forecast_bin, hist_errors_by_true_value_bin, ar_params
        )


def get_all_weather_params() -> List[str]:
    """
    Get a list of all weather parameters to fit AR(2) models for.

    Returns:
        list[str]: List of weather parameter names.
    """
    weather_params = config["main_params"].copy()
    for i, _ in enumerate(config["onshore_locations"]):
        for param in config["onshore_params"]:
            weather_params.append(f"onshore_{i}_{param}")
    for i, _ in enumerate(config["offshore_locations"]):
        for param in config["offshore_params"]:
            weather_params.append(f"offshore_{i}_{param}")
    return weather_params


def prepare_weather_forecast_errors(param: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the AR(2) model to historical forecast errors of the given weather parameter. The function uses a binning
    to estimate the conditional distribution of the forecast errors.

    Args:
        param (str): The weather parameter to process.

    Returns:
        np.ndarray: Historical values. Shape (n_forecast_dates * n_forecast_hours,).
        np.ndarray: Historical forecasts. Shape (n_forecast_dates, n_forecast_hours).
        np.ndarray: Historical forecast errors. Shape (n_forecast_dates, n_forecast_hours
    """
    cwd = Path(os.getcwd())
    path_data_meteo = cwd / config["path_data_meteo"]
    hist_forecast_errors = pd.read_csv(
        path_data_meteo / f"historical_forecast_error_{param}.csv",
        index_col=["time"],
        parse_dates=["time"],
    )

    # Get the historical forecasts
    if "onshore" in param:
        i = int(param.split("_")[1])
        weather_param = "_".join(param.split("_")[2:])
        historical_forecast_file = path_data_meteo / f"forecasts_onshore_{i}.csv"
    elif "offshore" in param:
        i = int(param.split("_")[1])
        weather_param = "_".join(param.split("_")[2:])
        historical_forecast_file = path_data_meteo / f"forecasts_offshore_{i}.csv"
    else:
        weather_param = param
        historical_forecast_file = path_data_meteo / "forecasts_main.csv"

    hist_time_series = pd.read_csv(
        historical_forecast_file,
        index_col=["time"],
        parse_dates=["time"],
    )[weather_param]

    n_forecast_dates = hist_forecast_errors.shape[0]

    # Convert to numpy array
    hist_forecast_errors = hist_forecast_errors.to_numpy()
    hist_forecasts = np.zeros_like(hist_forecast_errors)
    hist_values = np.zeros_like(hist_forecast_errors)

    # Calculate the total forecast value from the forecast_error and the historical value: forecast = real + error
    for forecast_date in np.arange(n_forecast_dates):
        hist_values[forecast_date, :] = hist_time_series[
            forecast_date * 24 : forecast_date * 24 + len(hist_forecast_errors[forecast_date, :])
        ]

        hist_forecasts[forecast_date, :] = (
            hist_forecast_errors[forecast_date, :] + hist_values[forecast_date, :]
        )

    return hist_values, hist_forecasts, hist_forecast_errors


if __name__ == "__main__":
    main()
