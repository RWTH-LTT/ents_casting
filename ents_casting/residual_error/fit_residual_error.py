"""Module to fit short-term AR(2) models for the GBMPredictionModel of the time series parameters."""

import copy
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ents_casting.config import config
from ents_casting.forecast_models.lightgbm_model import LgbmForecastModel
from ents_casting.forecast_models.demand import DemandModel
from ents_casting.forecast_models.price_el import PriceElModel
from ents_casting.forecast_models.emission_el import EmissionElModel
from ents_casting.forecast_models.pv_local import PVlocalModel
from ents_casting.ar_models.short_term_error import ShortTermArFitter


def main() -> None:
    """
    Main function to fit short term residual error models for all time series parameters.
    """
    for param in ["pv_local", "demand_el", "demand_heat", "demand_cold", "price_el", "emission_el"]:
        if param not in config["time_series_parameters"]:
            continue

        print(f"Fitting AR(2) model for {param} forecast errors.")
        training_year = config["training_years"][param]
        hist_values, hist_forecasts, hist_forecast_errors = compute_short_term_residual_errors(
            param=param, year=training_year
        )
        compute_error_metrics(hist_values, hist_forecasts, hist_forecast_errors)

        ar_model_fitter = ShortTermArFitter(param=param, n_bins=10)
        bin_edges, hist_errors_by_forecast_bin, hist_errors_by_true_value_bin, ar_params = (
            ar_model_fitter.fit_short_term_AR(hist_values, hist_forecasts, hist_forecast_errors)
        )
        ar_model_fitter.save_parameters(
            bin_edges, hist_errors_by_forecast_bin, hist_errors_by_true_value_bin, ar_params
        )


def instantiate_model(param: str) -> LgbmForecastModel:
    """Instantiate the appropriate GBMPredictionModel based on the time series parameter."""
    if param == "pv_local":
        return PVlocalModel()
    elif param in ["demand_el", "demand_heat", "demand_cold"]:
        return DemandModel(ts_param=param)
    elif param == "price_el":
        return PriceElModel()
    elif param == "emission_el":
        return EmissionElModel()
    else:
        raise ValueError(f"Unknown parameter: {param}")


def compute_short_term_residual_errors(param: str, year: int = 2023) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the residual errors for the given time series parameter. The residual error is the
    difference between the historical observations and forecasts made with forecast models that
    have perfect foresight on the future weather parameter.
    Args:
        param (str): The time series parameter to process.
        year (int): The year of which to compute the residual errors.

    Returns:
        np.ndarray: Historical values. Shape (n_forecast_dates * n_forecast_hours,).
        np.ndarray: Historical forecasts. Shape (n_forecast_dates, n_forecast_hours).
        np.ndarray: Historical forecast errors. Shape (n_forecast_dates, n_forecast_hours).
    """

    # Forecasts are made for each day of the year, until the forecast horizon is at the end of year
    n_forecast_hours = 144
    n_forecast_dates = 365 - n_forecast_hours // 24 - 168 // 24

    hist_values = np.zeros((n_forecast_dates, n_forecast_hours))
    hist_forecasts = np.zeros((n_forecast_dates, n_forecast_hours))
    hist_forecast_errors = np.zeros((n_forecast_dates, n_forecast_hours))

    # Instantiate the forecast model
    model = instantiate_model(param)

    # Load data for prediction model
    Y_data_complete, X_data = model.load_data(year)

    # Cutoff the first 168 hours due to lag features
    Y_data = copy.deepcopy(Y_data_complete)
    Y_data = Y_data[168:8760]
    for key in X_data:
        X_data[key] = X_data[key][168:8760]

    # Filter the parameters in X_data to only those used in the model
    X_data = {key: X_data[key] for key in model.input_parameters}

    # Convert to numpy arrays for easier indexing
    y = np.asarray(Y_data)
    X = np.asarray(list(X_data.values())).T

    # Start with forecast day 7 to have enough lag values available
    n_days_per_fold = config["n_days_per_cross_validation_fold"]
    for forecast_date in range(int(n_forecast_dates)):
        # Load the pretrained fold model at the same fold boundary used during training
        if forecast_date % n_days_per_fold == 0:
            fold_idx = forecast_date // n_days_per_fold
            model.load_gbm_model(fold=fold_idx)

        # Create a forecast for each forecast day
        forecast_start_idx = forecast_date * 24
        forecast_end_idx = forecast_start_idx + n_forecast_hours

        X_forecast_data = {p: X_data[p][forecast_start_idx:forecast_end_idx] for p in model.input_parameters}
        lag_values = list(Y_data_complete[forecast_start_idx : forecast_start_idx + 168])

        Y_forecast_pred = model.predict_with_lag(X_forecast_data, lag_values=lag_values)

        # Ensure no negative values
        if param != "price_el":
            Y_forecast_pred = Y_forecast_pred.clip(min=0.0)

        Y_forecast_true = Y_data[forecast_start_idx:forecast_end_idx]
        hist_values[forecast_date, :] = Y_forecast_true
        hist_forecasts[forecast_date, :] = Y_forecast_pred
        hist_forecast_errors[forecast_date, :] = Y_forecast_true - Y_forecast_pred

    return hist_values, hist_forecasts, hist_forecast_errors


def compute_error_metrics(
    hist_values: np.ndarray, hist_forecasts: np.ndarray, hist_forecast_errors: np.ndarray
) -> None:
    """
    Args:
        hist_values (np.ndarray): Historical values. Shape (n_forecast_dates, n_forecast_hours).
        hist_forecasts (np.ndarray): Historical forecasts. Shape (n_forecast_dates, n_forecast_hours).
        hist_forecast_errors (np.ndarray): Historical forecast errors. Shape (n_forecast_dates, n_forecast_hours).

    """

    r2 = 1 - np.sum(hist_forecast_errors.flatten() ** 2) / np.sum((hist_values - np.mean(hist_values)) ** 2)
    mae = mean_absolute_error(hist_values.flatten(), hist_forecasts.flatten())
    rmse = np.sqrt(mean_squared_error(hist_values.flatten(), hist_forecasts.flatten()))

    print("Residual error metrics (short-term forecasts with perfect weather forecasts):")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


if __name__ == "__main__":
    main()
