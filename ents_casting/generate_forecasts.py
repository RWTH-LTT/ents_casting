"""
Precomputed the forecasts made at the start of each day of a year. Considers errors in weather
forecasts and residual errors in the prediction models.
"""

import copy
import itertools
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

from ents_casting.config import config
from ents_casting.forecast_models.demand import DemandModel
from ents_casting.forecast_models.price_el import PriceElModel
from ents_casting.forecast_models.emission_el import EmissionElModel
from ents_casting.forecast_models.pv_local import PVlocalModel
from ents_casting.forecast_models.wind_local import WindLocalModel
from ents_casting.ar_models.short_term_error import ErrorGenerator
from ents_casting.generate_long_term_scenarios import (
    extrapolate_weather_trends,
    load_additional_data,
    load_weather_data,
)
from ents_casting.weather_forecast_error.fit_forecast_error import get_all_weather_params


def main(n_forecasts: int = config["n_forecasts"], years: list[int] = []) -> None:
    # Load all input data for forecast models
    weather_data = load_weather_data()
    weather_data = extrapolate_weather_trends(weather_data)
    additional_data = load_additional_data(weather_data["datetime_local"])

    # Load long-term scenarios to access lagged values
    path = Path.cwd() / config["path_data_long_term_scenarios"]
    long_term_scenarios = pd.read_csv(path / "long_term_scenarios.csv")

    output_path = Path.cwd() / config["path_data_forecasts"]

    # Generate perfect time series forecasts for each year
    if years == []:
        years = long_term_scenarios["year"].unique().tolist()
    for year in years:
        perfect_forecasts = get_perfect_time_series_forecast(long_term_scenarios[long_term_scenarios["year"] == year])
        # Save forecasts
        np.savez(
            output_path / f"perfect_forecasts_{year}.npz",
            **perfect_forecasts,
        )

    # Generate stochastic time series forecasts for each year
    for year in years:
        stoch_forecasts = precompute_forecasts_for_year(
            weather_data[weather_data["year"] == year],
            long_term_scenarios[long_term_scenarios["year"] == year],
            additional_data[additional_data["year"] == year],
            n_forecasts,
        )
        # Save forecasts
        np.savez(
            output_path / f"stochastic_forecasts_{year}_{n_forecasts}.npz",
            **stoch_forecasts,
        )
        print(f"Saved stochastic forecasts for year {year} to {output_path}")


def precompute_forecasts_for_year(
    weather_data: pd.DataFrame,
    future_year_scenario: pd.DataFrame,
    additional_data: pd.DataFrame,
    n_forecasts: int,
) -> None:
    print(f"Precomputing stochastic forecasts for year {weather_data['year'].iloc[0]}")

    perfect_weather_forecast = get_perfect_weather_forecast(weather_data)
    det_weather_forecast = generate_synthetic_deterministic_forecasts(perfect_weather_forecast)
    stoch_weather_forecasts = generate_stochastic_weather_forecasts(det_weather_forecast, n_forecasts)
    stoch_forecasts = generate_stochastic_forecasts(stoch_weather_forecasts, additional_data, future_year_scenario)
    return stoch_forecasts


def get_perfect_time_series_forecast(
    future_year_scenario: pd.DataFrame, forecast_horizon: int = 144
) -> Dict[str, np.ndarray]:
    """
    Generate the perfect forecast for each day, ensuring a fixed forecast length.
    Missing values at the end of the forecast are filled with 0.

    Args:
        future_year_scenario (pd.DataFrame): DataFrame containing the data for one long-term.

    Returns:
        dict: A dict, where the key is the parameter (str), and the value is a A 3D array with dimensions
            (365 days x 1 scenario x 144 forecast hours).
    """
    perfect_forecasts = {}
    for param in future_year_scenario.columns:
        if param in ["datetime_local", "year", "hour_in_year"]:
            continue
        perfect_forecasts[param] = np.zeros((365, 1, forecast_horizon), dtype=np.float32)
        param_values = future_year_scenario[param].values
        for day in range(365):
            forecast_start = day * 24
            forecast_end = min(8760, day * 24 + forecast_horizon)
            forecast_length = forecast_end - forecast_start
            perfect_forecasts[param][day, 0, :forecast_length] = param_values[forecast_start:forecast_end]

    return perfect_forecasts


def get_perfect_weather_forecast(weather_data: pd.DataFrame, forecast_horizon: int = 144) -> Dict[str, np.ndarray]:
    """
    Generate the perfect forecast for each day, ensuring a fixed forecast length.
    Missing values at the end of the forecast are filled with 0.

    Args:
        weather_data (pd.DataFrame): DataFrame containing the weather data for one year.

    Returns:
        dict: A dict, where the key is the parameter (str), and the value is a A 3D array with dimensions
            (365 days x 1 scenario x 144 forecast hours).
    """
    weather_params = get_all_weather_params()

    perfect_forecasts = {}
    for param in weather_params:
        if param in ["datetime_local", "year", "hour_in_year"]:
            continue
        perfect_forecasts[param] = np.zeros((365, 1, forecast_horizon), dtype=np.float32)
        param_values = weather_data[param].values
        for day in range(365):
            forecast_start = day * 24
            forecast_end = min(8760, day * 24 + forecast_horizon)
            forecast_length = forecast_end - forecast_start
            perfect_forecasts[param][day, 0, :forecast_length] = param_values[forecast_start:forecast_end]

    return perfect_forecasts


def generate_synthetic_deterministic_forecasts(
    perfect_weather_forecast: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic deterministic weather forecasts from perfect input data.

    Args:
        perfect_input_data (dict): A dict where the key is the parameter (str), and the value is a 3D array
            with dimensions (365 days x 1 scenario x forecast_horizon).
        forecast_horizon (int): Number of hours to forecast ahead for each day.
    """
    det_weather_forecast = copy.deepcopy(perfect_weather_forecast)
    weather_params = get_all_weather_params()
    for param in weather_params:
        # Instantiate the forecast error model
        ar_model = ErrorGenerator(param=param)
        for day in range(perfect_weather_forecast[param].shape[0]):
            error_series = ar_model.generate_reverse_error_series(
                true_values=perfect_weather_forecast[param][day, 0, :]
            )
            det_weather_forecast[param][day] -= error_series

        # Ensure no negative values
        if "temperature" not in param:
            det_weather_forecast[param] = det_weather_forecast[param].clip(min=0.0)

    return det_weather_forecast


def generate_stochastic_weather_forecasts(
    det_weather_forecast: Dict[str, np.ndarray],
    n_scenarios: int,
) -> Dict[str, np.ndarray]:
    """
    Generate stochastic weather forecasts from deterministic weather forecasts.

    Args:
        det_weather_forecast (dict): A dict where the key is the parameter (str), and the value is a 3D array
            with dimensions (365 days x 1 scenario x forecast_horizon).
        n_scenarios (int): Number of stochastic scenarios to generate.
    Returns:
        dict: A dict where the key is the parameter (str), and the value is a 3D array with dimensions
            (365 days x n_scenarios x forecast_horizon).
    """
    stoch_weather_forecast = {}
    n_forecast_days = det_weather_forecast[list(det_weather_forecast.keys())[0]].shape[0]
    forecast_horizon = det_weather_forecast[list(det_weather_forecast.keys())[0]].shape[2]

    weather_params = get_all_weather_params()
    for param in weather_params:
        ar_model = ErrorGenerator(param=param)
        det_forecast = det_weather_forecast[param][:, 0, :]
        stoch_weather_forecast[param] = np.zeros(
            (n_forecast_days, n_scenarios, forecast_horizon),
            dtype=np.float32,
        )
        for scenario in range(n_scenarios):
            for day in range(n_forecast_days):
                if scenario == 0:  # first scenario is the deterministic forecast without additional error
                    error_series = np.zeros(forecast_horizon, dtype=np.float32)
                else:
                    error_series = ar_model.generate_error_series(forecast=det_forecast[day])
                stoch_weather_forecast[param][day, scenario, :] = det_forecast[day] + error_series

            # Ensure no negative values, unless temperature
            if "temperature" not in param:
                stoch_weather_forecast[param] = stoch_weather_forecast[param].clip(min=0.0)

    return stoch_weather_forecast


def generate_stochastic_forecasts(
    stoch_weather_forecasts: Dict[str, np.ndarray],
    additional_data: pd.DataFrame,
    future_year_scenario: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate stochastic forecasts of time series parameters using stochastic weather forecasts.

    Args:
        stoch_weather_forecasts (dict): A dict where the key is the weather parameter (str), and the value is a 3D array
            with dimensions (365 days x n_scenarios x forecast_horizon).
        future_year_scenario (pd.DataFrame): DataFrame containing the long-term scenario data for lagged values.

    Returns:
        dict: A dict where the key is the time series parameter (str), and the value is another dict with keys as
            scenario indices (str) and values as 3D arrays with dimensions (365 days x 1 scenario x forecast_horizon).
    """
    # Create data structure to hold stochastic forecasts
    n_forecast_days = stoch_weather_forecasts[list(stoch_weather_forecasts.keys())[0]].shape[0]
    n_scenarios = stoch_weather_forecasts[list(stoch_weather_forecasts.keys())[0]].shape[1]
    forecast_horizon = stoch_weather_forecasts[list(stoch_weather_forecasts.keys())[0]].shape[2]
    ts_params = [
        "wind_local",
        "pv_local",
        "demand_el",
        "demand_heat",
        "demand_cold",
        "price_el",
        "emission_el",
    ]
    stoch_forecasts = {
        param: np.zeros((n_forecast_days, n_scenarios, forecast_horizon), dtype=np.float32) for param in ts_params
    }

    # Load all models
    wind_model = WindLocalModel()
    gbm_models = {
        "pv_local": PVlocalModel(),
        "demand_el": DemandModel("demand_el"),
        "demand_heat": DemandModel("demand_heat"),
        "demand_cold": DemandModel("demand_cold"),
        "price_el": PriceElModel(),
        "emission_el": EmissionElModel(),
    }
    # Remove models for parameters not in config
    gbm_models = {param: model for param, model in gbm_models.items() if param in config["time_series_parameters"]}

    ar_models = {
        param: ErrorGenerator(param=param) for param in config["time_series_parameters"] if param != "wind_local"
    }

    n_days_per_fold = config["n_days_per_cross_validation_fold"]
    fold_length = n_days_per_fold * 24

    # Generate forecasts for each forecast_day and scenario
    for forecast_day in range(n_forecast_days):
        current_ts = forecast_day * 24
        fold_idx = current_ts // fold_length

        for model in gbm_models.values():
            model.load_gbm_model(fold=fold_idx)

        for scenario in range(n_scenarios):
            time_steps_to_end_of_year = [ts for ts in range(current_ts, current_ts + forecast_horizon) if ts < 8760]
            # Extract forecast for the current day and scenario
            forecast = {
                param: stoch_weather_forecasts[param][forecast_day, scenario, :] for param in stoch_weather_forecasts
            }

            # Add additional data to the forecast
            for param in additional_data.columns:
                if param in ["datetime_local", "year", "hour_in_year"]:
                    continue
                forecast[param] = np.zeros(forecast_horizon, dtype=np.float32)
                forecast[param][: len(time_steps_to_end_of_year)] = additional_data[param].values[
                    time_steps_to_end_of_year
                ]

            # Forecast wind_local
            stoch_forecasts["wind_local"][forecast_day, scenario, :] = wind_model.predict(
                data=forecast,
            )

            # Forecast other parameters
            for param, model in gbm_models.items():
                # Get lag values from long-term scenario
                lag_start_ts = current_ts - 168
                if lag_start_ts < 0:
                    # Need to wrap around: take remaining values from end of the year
                    lag_values = list(future_year_scenario[param][lag_start_ts:])  # from end
                    lag_values.extend(future_year_scenario[param][:current_ts])  # from start
                else:
                    lag_values = list(future_year_scenario[param][lag_start_ts:current_ts])

                stoch_forecasts[param][forecast_day, scenario, :] = model.predict_with_lag(
                    data=forecast,
                    lag_values=lag_values,
                )

                if scenario > 0:
                    # Add residual error
                    stoch_forecasts[param][forecast_day, scenario, :] += ar_models[param].generate_error_series(
                        forecast=stoch_forecasts[param][forecast_day, scenario, :]
                    )

                # Ensure no negative values
                if param != "price_el":
                    stoch_forecasts[param][forecast_day, scenario, :] = stoch_forecasts[param][
                        forecast_day, scenario, :
                    ].clip(min=0.0)

                # Ensure that forecasted pv_local is 0 during night time (solar elevation >= 90 degrees)
                if param == "pv_local":
                    stoch_forecasts[param][forecast_day, scenario, :][forecast["apparent_zenith"] >= 90] = 0.0

    return stoch_forecasts


if __name__ == "__main__":
    # Parse console arguments and run main function
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_forecasts",
        type=int,
        default=5,
        help="Number of stochastic forecast scenarios to generate.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year to generate forecasts for.",
    )
    args = parser.parse_args()
    n_forecasts = args.n_forecasts
    year = args.year

    main(n_forecasts=n_forecasts, years=[year])
