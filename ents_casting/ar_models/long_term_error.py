"""
Module for fitting and generating long-term AR(2) models for residual forecast errors. Long-term
AR(2) models capture the dependency of forecast errors on the forecasted value and the daytime.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import optimize
import pickle
from scipy.stats import norm

from ents_casting.config import config


class LongTermArFitter:
    """
    Attributes:
        param (str): The parameter to predict.
        path_data_ar_model_params (Path): Path to the file where the fitted AR(2) model parameters are saved.
        n_bins (int): Number of bins used for the historical forecast errors.
    """

    def __init__(self, param: str, n_bins: int) -> None:
        """
        Initialize the ARModelFitter.

        Args:
            param (str): The parameter to predict.
            n_bins (int): Number of bins used for the historical forecast errors.
        """
        self.param = param
        self.path_data_ar_model_params = Path(os.getcwd()) / config["path_data_ar_model_params"]
        self.n_bins = n_bins

    def fit_long_term_AR(
        self,
        hist_values: pd.DataFrame,
        hist_forecasts: np.ndarray,
        hist_forecast_errors: np.ndarray,
    ) -> tuple[np.ndarray, dict[(int, int), np.ndarray], list[float]]:
        """
        Fit an AR(2) model to the historical forecast errors.

        Args:
            hist_values (pd.Series): Historical values of the parameter.
            hist_forecasts (np.ndarray): Historical forecasts of the parameter. Shape: (n_forecast_hours).
            hist_forecast_errors (np.ndarray): Historical forecast errors. Shape: (n_forecast_hours).

        Returns:
            tuple[np.ndarray, dict[(int, int), np.ndarray], list[float]]:
                - bin_edges (np.ndarray): The edges of the bins used for the forecast values.
                - hist_errors_by_forecast_bin (dict[(int, int), np.ndarray]): Sorted historical forecast errors for each bin and daytime.
                - ar_params (list[float]): Fitted AR(2) model parameters: [c, phi_1, phi_2, sigma].
        """
        n_daytimes = 24
        n_forecast_hours = hist_forecasts.shape[0]

        # Assign bin to each forecast value
        bin_edges = np.linspace(hist_values.min(), hist_values.max(), self.n_bins + 1)
        bin_by_forecast_value = np.digitize(hist_forecasts, bin_edges, right=False) - 1
        bin_by_forecast_value = np.clip(bin_by_forecast_value, 0, self.n_bins - 1)

        # Collect the distribution of forecast errors for each bin and daytime
        hist_errors_by_forecast_bin = {}
        for daytime in np.arange(n_daytimes):
            # Collect all time steps for the current daytime, assuming that hist_forecast_errors is hourly data and starts at hour 0
            hist_forecast_errors_daytime = hist_forecast_errors[daytime::n_daytimes]
            for bin in np.arange(self.n_bins):
                # Filter the forecast errors that are for the current bin
                indices = np.where(bin_by_forecast_value[daytime::n_daytimes] == bin)[0]
                hist_forecast_errors_filtered = hist_forecast_errors_daytime[indices]
                hist_forecast_errors_filtered.sort()
                hist_errors_by_forecast_bin[daytime, bin] = hist_forecast_errors_filtered

        # Map the historical errors to a normal distribution, based on the bins and the sorted errors
        hist_forecast_errors_normal_distr = {0: np.zeros(n_forecast_hours)}
        for time_step in np.arange(n_forecast_hours):
            daytime = time_step % n_daytimes
            # Get absolute error
            error = hist_forecast_errors[time_step]

            # Get the bin for the current forecast day and lead time
            bin = bin_by_forecast_value[time_step]

            # Get the sorted historical forecast errors for the current bin
            hist = hist_errors_by_forecast_bin[daytime, bin]

            # Calculate the rank and percentile value
            rank = np.interp(error, hist, np.arange(len(hist)))
            percentile_value = rank / len(hist)
            percentile_value = np.clip(percentile_value, 0.001, 0.999)

            # Map to normal distribution
            normalized_error = norm.ppf(percentile_value, loc=0, scale=1)
            hist_forecast_errors_normal_distr[0][time_step] = normalized_error

        # Fit AR(2) model with bias using the Expectation Maximization algorithm
        c, phi_1, phi_2, sigma2 = self._fit_ar2_parameters(hist_forecast_errors_normal_distr)
        ar_params = [c, phi_1, phi_2, np.sqrt(sigma2)]

        print(f"AR(2) parameters: c={c}, phi_1={phi_1}, phi_2={phi_2}, sigma2={sigma2}")

        return bin_edges, hist_errors_by_forecast_bin, ar_params

    def save_parameters(self, bin_edges, hist_errors_by_forecast_bin, ar_params) -> None:
        """
        Save all required model parameters.
        """
        self._save_bin_edges(bin_edges)
        self._save_errors_by_forecast_bin(hist_errors_by_forecast_bin)
        self._save_ar_params(ar_params)

    def _fit_ar2_parameters(self, data: dict[int, np.ndarray]) -> tuple[float, float, float, float]:
        """
        Fits an AR(2) model with bias to the given data using the Expectation-Maximization (EM) algorithm.

        Args:
            data (dict[int, np.ndarray]): A dictionary where the keys are indices (e.g., time steps) and the values
                are numpy arrays representing the time series data for each index.

        Returns:
            tuple[float, float, float, float]: A tuple containing the fitted parameters:
                - c (float): The bias term.
                - phi_1 (float): The first autoregressive coefficient.
                - phi_2 (float): The second autoregressive coefficient.
                - sigma2 (float): The variance of the residuals.
        """
        result = optimize.differential_evolution(
            self._log_likelihood_AR2, bounds=[(-1, 1), (-1, 1), (-1, 1)], args=(data,)
        )
        c, phi_1, phi_2 = result.x
        residuals = self._calculate_AR2_residuals(data, c, phi_1, phi_2)
        sigma2 = np.var(residuals)

        return c, phi_1, phi_2, sigma2

    def _log_likelihood_AR2(self, params: tuple[float, float, float], data: dict[int, np.ndarray]) -> float:
        """
        Computes the negative log-likelihood of an AR(2) model with bias for the given data.

        Args:
            params (tuple[float, float, float, float]): A tuple containing the AR(2) model parameters:
                - c (float): The bias term.
                - phi_1 (float): The first autoregressive coefficient.
                - phi_2 (float): The second autoregressive coefficient.
                - sigma2 (float): The variance of the residuals.
            data (dict[int, np.ndarray]): A dictionary where the keys are indices (e.g., time steps) and the values
                are numpy arrays representing the time series data for each index.

        Returns:
            float: The negative log-likelihood of the AR(2) model for the given data.
        """
        c, phi_1, phi_2 = params

        # Calculate the residuals
        residuals = self._calculate_AR2_residuals(data, c, phi_1, phi_2)

        # Calculate the variance of the residuals
        sigma2 = np.var(residuals)

        # Return the negative log-likelihood, as scipy.optimize.minimize performs minimization
        log_likelihood = -0.5 * len(residuals) * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2
        return -log_likelihood

    def _calculate_AR2_residuals(self, data, c, phi_1, phi_2):
        residuals = []
        for series in data.values():
            for t in range(2, len(series)):
                residual = series[t] - c - phi_1 * series[t - 1] - phi_2 * series[t - 2]
                residuals.append(residual)
        return np.array(residuals)

    def _save_bin_edges(self, bin_edges) -> None:
        """
        Save the bin edges to a pickle file.
        """
        with open(self.path_data_ar_model_params / f"{self.param}_long_term_bin_edges.pickle", "wb") as f:
            pickle.dump(bin_edges, f)

    def _save_errors_by_forecast_bin(self, hist_errors_by_forecast_bin) -> None:
        """
        Save the sorted histogram of forecast errors binned by forecast value to a pickle file.
        """
        with open(
            self.path_data_ar_model_params / f"{self.param}_long_term_hist_errors_by_forecast_bin.pickle",
            "wb",
        ) as g:
            pickle.dump(hist_errors_by_forecast_bin, g)

    def _save_ar_params(self, ar_params) -> None:
        """
        Save the AR(2) model parameters to a pickle file.
        """
        with open(self.path_data_ar_model_params / f"{self.param}_long_term_ar_params.pickle", "wb") as f:
            pickle.dump(ar_params, f)


class ErrorGenerator:
    """
    This class generates forecast errors for a given parameter using the fitted AR(2) model. It
    denormalizes the generated errors with the empirical distribution of historical forecast errors
    depending on the forecasted value bins and the lead time.

    Attributes:
        param (str): The parameter to predict.
        hist_errors_by_forecast_bin (dict[(int,int), np.ndarray]): Sorted historical forecast errors.
        ar_params (list[float]): Fitted AR(2) model parameters.
        bin_edges (np.ndarray): Bin edges for the forecast values.
        n_bins (int): Number of bins.
    """

    def __init__(self, param: str) -> None:
        """
        Initialize the ARModelGenerator.

        Args:
            param (str): The parameter to predict.
        """
        self.param = param
        self.path_data_ar_model_params = Path(os.getcwd()) / config["path_data_ar_model_params"]

        self.hist_errors_by_forecast_bin = self._load_hist_forecast_errors()
        self.ar_params = self._load_ar_params()
        self.bin_edges = self._load_bin_edges()
        self.n_bins = len(self.bin_edges) - 1

    def generate_error_series(self, forecast: np.ndarray[float]) -> np.ndarray:
        """
        Generates a single forecast error series using the fitted AR(2) model.
        The generated series is denormalized to match the historical data distribution.

        Args:
            forecast (np.ndarray[float]): The forecast values with the lenght of forecast_horizon.
            hours_in_year (np.ndarray[int]): Array of hours in the year corresponding to the forecast times.

        Returns:
            np.ndarray: The generated forecast error series.
        """
        forecast_horizon = len(forecast)

        # Generate an AR(2) error series
        ar_series = self._generate_ar_series(forecast_horizon)

        # Determine the bin for each forecast value
        bin_by_lookahead = np.digitize(forecast, self.bin_edges, right=False) - 1
        bin_by_lookahead = np.clip(bin_by_lookahead, 0, self.n_bins - 1)

        # Denormalize the AR(2) error series with hist_errors_by_forecast_bin
        denormalized_error_series = self._map_to_empirical_distribution(ar_series, bin_by_lookahead)

        return denormalized_error_series

    def _load_hist_forecast_errors(self) -> dict[(int, int), np.ndarray]:
        """
        Load the sorted histogram of forecast errors from a pickle file.

        Returns:
            dict[(int, int), np.ndarray]: A dictionary where the keys are lead times and the values are sorted forecast errors.
        """
        with open(
            self.path_data_ar_model_params / f"{self.param}_long_term_hist_errors_by_forecast_bin.pickle",
            "rb",
        ) as g:
            hist_errors_by_forecast_bin = pickle.load(g)
        return hist_errors_by_forecast_bin

    def _load_bin_edges(self) -> np.ndarray:
        """
        Load the bin edges from a pickle file.

        Returns:
            np.ndarray: The bin edges.
        """
        with open(self.path_data_ar_model_params / f"{self.param}_long_term_bin_edges.pickle", "rb") as f:
            bin_edges = pickle.load(f)
        return bin_edges

    def _load_ar_params(self) -> list[float]:
        """
        Load the AR(2) model parameters from a pickle file.

        Returns:
            list[float]: A list containing the AR(2) model parameters: c, phi_1, phi_2, sigma2.
        """
        with open(self.path_data_ar_model_params / f"{self.param}_long_term_ar_params.pickle", "rb") as f:
            ar_params = pickle.load(f)
        return ar_params

    def _generate_ar_series(self, forecast_horizon: int) -> np.ndarray:
        """
        Generate an AR(2) error series.

        Args:
            forecast_horizon (int): The forecast horizon (number of lead times).

        Returns:
            np.ndarray: The generated AR(2) error series.
        """
        c, phi_1, phi_2, sigma = self.ar_params
        ar_series = np.zeros(forecast_horizon)
        for lead_time in range(forecast_horizon):
            if lead_time == 0:
                ar_series[lead_time] = np.random.normal(0, sigma)
            elif lead_time == 1:
                ar_series[lead_time] = c + phi_1 * ar_series[lead_time - 1] + np.random.normal(0, sigma)
            else:
                ar_series[lead_time] = (
                    c + phi_1 * ar_series[lead_time - 1] + phi_2 * ar_series[lead_time - 2]
                ) + np.random.normal(0, sigma)

        return ar_series

    def _map_to_empirical_distribution(
        self, ar_series: np.ndarray[float], bin_by_lookahead: np.ndarray[int]
    ) -> np.ndarray:
        """
        Map the AR(2) error series to the empirical distribution.

        Args:
            ar_series (np.ndarray): The AR(2) error series.
            bin_by_lookahead (np.ndarray): The bin corresponding to the forecast values.

        Returns:
            np.ndarray: The denormalized error series.
        """
        forecast_horizon = len(ar_series)
        norm_cdf_percentile_series = norm.cdf(ar_series, loc=0, scale=1)

        # Map the errors based on the percentiles in self.hist_errors_by_forecast_bin
        denormalized_error_series = np.zeros(forecast_horizon)
        for lead_time in range(forecast_horizon):
            day_time = lead_time % 24
            bin = bin_by_lookahead[lead_time]
            # Retrieve sorted historical forecast error values
            if len(self.hist_errors_by_forecast_bin[day_time, bin]) != 0:
                sorted_hist_forecast_error = self.hist_errors_by_forecast_bin[day_time, bin]
            else:
                sorted_hist_forecast_error = np.array([0])

            denormalized_error_series[lead_time] = np.interp(
                norm_cdf_percentile_series[lead_time],
                np.linspace(0, 1, len(sorted_hist_forecast_error)),
                sorted_hist_forecast_error,
            )
            # check for nan
            if np.isnan(denormalized_error_series[lead_time]):
                print(f"NaN value in denormalized_error_series at lead time {lead_time}")

        return denormalized_error_series
