"""
Module for fitting and generating short-term AR(2) models for weather forecast errors and
residual forecast errors. Short-term AR(2) models capture the dependency of forecast errors on
the forecasted value and the lead time.
"""

import numpy as np
import os
from pathlib import Path
from scipy import optimize
import pickle
from scipy.stats import norm

from ents_casting.config import config


class ShortTermArFitter:
    """
    This class fits an AR(2) model to the forecast errors of a given parameter. It uses the probability integral
    transformation based on the bin of the forecasted value and the lead time to map the errors to a normal distribution.
    Then, it fits the AR(2) model to the normalized errors using the Expectation Maximization (EM) algorithm.

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

    def fit_short_term_AR(
        self,
        hist_values: np.ndarray,
        hist_forecasts: np.ndarray,
        hist_forecast_errors: np.ndarray,
    ) -> tuple[np.ndarray, dict[(int, int), np.ndarray], dict[(int, int), np.ndarray], list[float]]:
        """
        Fit an AR(2) model to the historical forecast errors.

        Args:
            hist_values (np.ndarray): Historical values of the parameter. Shape: (n_forecast_dates, n_lead_times).
            hist_forecasts (np.ndarray): Historical forecasts of the parameter. Shape: (n_forecast_dates, n_lead_times).
            hist_forecast_errors (np.ndarray): Historical forecast errors. Shape: (n_forecast_dates, n_lead_times).

        Returns:
            tuple[np.ndarray, dict[(int, int), np.ndarray], dict[(int, int), np.ndarray], list[float]]:
                - bin_edges (np.ndarray): The edges of the bins used for both forecast and true values.
                - hist_errors_by_forecast_bin (dict[(int, int), np.ndarray]): Sorted historical forecast errors for each bin of the forecasted value and each lead time.
                - hist_errors_by_true_value_bin (dict[(int, int), np.ndarray]): Sorted historical forecast errors for each bin of the true value and each lead time.
                - ar_params (list[float]): Fitted AR(2) model parameters: [c, phi_1, phi_2, sigma].
        """
        n_lead_times = hist_forecast_errors.shape[1]
        n_forecast_dates = hist_forecast_errors.shape[0]

        # Assign bin to each forecast value
        bin_edges = np.linspace(hist_values.min(), hist_values.max(), self.n_bins + 1)
        bin_by_forecast_value = np.digitize(hist_forecasts, bin_edges, right=False) - 1
        bin_by_forecast_value = np.clip(bin_by_forecast_value, 0, self.n_bins - 1)
        bin_by_true_value = np.digitize(hist_values, bin_edges, right=False) - 1
        bin_by_true_value = np.clip(bin_by_true_value, 0, self.n_bins - 1)

        # Collect the distribution of forecast errors for each bin and lead time
        hist_errors_by_forecast_bin = {}
        hist_errors_by_true_value_bin = {}
        for lead_time in np.arange(n_lead_times):
            errors_for_lead_time = hist_forecast_errors[:, lead_time].copy()
            for bin in np.arange(self.n_bins):
                # Filter the forecast errors that are for the current bin
                indices = np.where(bin_by_forecast_value[:, lead_time] == bin)[0]
                errors = errors_for_lead_time[indices]
                errors.sort()
                hist_errors_by_forecast_bin[lead_time, bin] = errors

                indices_true = np.where(bin_by_true_value[:, lead_time] == bin)[0]
                errors_true = errors_for_lead_time[indices_true]
                errors_true.sort()
                hist_errors_by_true_value_bin[lead_time, bin] = errors_true

        # Map the historical errors to a normal distribution, based on the bins and the sorted errors
        hist_forecast_errors_normal_distr = {}
        for forecast_day in np.arange(n_forecast_dates):
            normalized_errors = []
            for lead_time in np.arange(n_lead_times):
                # Skip lead_times for which low solar is forecasted
                if (
                    self.param in ["direct_normal_irradiance", "shortwave_radiation"]
                    and hist_values[forecast_day, lead_time] <= 20
                ):
                    continue
                elif self.param == "pv_local" and hist_values[forecast_day, lead_time] <= 0.05:
                    continue

                # Get absolute error
                error = hist_forecast_errors[forecast_day, lead_time]

                # Get the bin for the current forecast day and lead time
                bin = bin_by_forecast_value[forecast_day, lead_time]

                # Get the sorted historical forecast errors for the current bin
                hist = hist_errors_by_forecast_bin[lead_time, bin]
                if hist is None:
                    normalized_errors.append(0.0)
                else:
                    # Calculate the rank and percentile value
                    rank = np.interp(error, hist, np.arange(len(hist)))
                    percentile_value = rank / len(hist)
                    percentile_value = np.clip(percentile_value, 0.001, 0.999)

                    # Map to normal distribution
                    normalized_error = norm.ppf(percentile_value, loc=0, scale=1)
                    normalized_errors.append(normalized_error)
            hist_forecast_errors_normal_distr[forecast_day] = np.array(normalized_errors)

        # Fit AR(2) model with bias using the Expectation Maximization algorithm
        c, phi_1, phi_2, sigma2 = self._fit_ar2_parameters(hist_forecast_errors_normal_distr)
        ar_params = [c, phi_1, phi_2, np.sqrt(sigma2)]

        print(
            f"Fitted AR(2) parameters for {self.param}: c={c}, phi_1={phi_1}, phi_2={phi_2}, sigma2={sigma2}"
        )

        return bin_edges, hist_errors_by_forecast_bin, hist_errors_by_true_value_bin, ar_params

    def save_parameters(
        self, bin_edges, hist_errors_by_forecast_bin, hist_errors_by_true_value_bin, ar_params
    ) -> None:
        """
        Save all required model parameters.
        """
        self._save_bin_edges(bin_edges)
        self._save_errors_by_forecast_bin(hist_errors_by_forecast_bin)
        self._save_errors_by_true_value_bin(hist_errors_by_true_value_bin)
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

    def _log_likelihood_AR2(
        self, params: tuple[float, float, float], data: dict[int, np.ndarray]
    ) -> float:
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
        if sigma2 <= 0:
            print("Warning: Non-positive variance encountered in log-likelihood calculation.")

        # Return the negative log-likelihood, as scipy.optimize.minimize performs minimization
        log_likelihood = (
            -0.5 * len(residuals) * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2
        )
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
        with open(self.path_data_ar_model_params / f"{self.param}_bin_edges.pickle", "wb") as f:
            pickle.dump(bin_edges, f)

    def _save_errors_by_forecast_bin(self, hist_errors_by_forecast_bin) -> None:
        """
        Save the sorted histogram of forecast errors binned by forecast value to a pickle file.
        """
        with open(
            self.path_data_ar_model_params / f"{self.param}_hist_errors_by_forecast_bin.pickle",
            "wb",
        ) as g:
            pickle.dump(hist_errors_by_forecast_bin, g)

    def _save_errors_by_true_value_bin(self, hist_errors_by_true_value_bin) -> None:
        """
        Save the sorted histogram of forecast errors binned by the true value to a pickle file.
        """
        with open(
            self.path_data_ar_model_params / f"{self.param}_hist_errors_by_true_value_bin.pickle",
            "wb",
        ) as g:
            pickle.dump(hist_errors_by_true_value_bin, g)

    def _save_ar_params(self, ar_params) -> None:
        """
        Save the AR(2) model parameters to a pickle file.
        """
        with open(self.path_data_ar_model_params / f"{self.param}_ar_params.pickle", "wb") as f:
            pickle.dump(ar_params, f)


class ErrorGenerator:
    """
    This class generates forecast errors for a given parameter using the fitted AR(2) model. It
    denormalizes the generated errors with the empirical distribution of historical forecast errors
    depending on the forecasted value bins and the lead time.

    Attributes:
        param (str): The parameter to predict.
        errors_by_forecast_bin (dict[(int,int), np.ndarray]): Sorted historical forecast errors.
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

        self.errors_by_forecast_bin = self._load_errors_by_forecast_bin()
        self.errors_by_true_value_bin = self._load_errors_by_true_value_bin()
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

        # Denormalize the AR(2) error series with errors_by_forecast_bin
        denormalized_error_series = self._map_to_empirical_distribution(
            ar_series, bin_by_lookahead, self.errors_by_forecast_bin
        )

        return denormalized_error_series

    def generate_reverse_error_series(self, true_values: np.ndarray[float]) -> np.ndarray:
        """
        Generates a single forecast error series using the fitted AR(2) model.
        The generated series is denormalized to match the historical data distribution.
        This method uses the true values to determine the bins instead of the forecasted values.

        Args:
            true_values (np.ndarray[float]): The true values with the length of forecast_horizon.
        Returns:
            np.ndarray: The generated forecast error series.
        """
        forecast_horizon = len(true_values)

        # Generate an AR(2) error series
        ar_series = self._generate_ar_series(forecast_horizon)

        # Determine the bin for each forecast value
        bin_by_lookahead = np.digitize(true_values, self.bin_edges, right=False) - 1
        bin_by_lookahead = np.clip(bin_by_lookahead, 0, self.n_bins - 1)

        # Denormalize the AR(2) error series with errors_by_forecast_bin
        denormalized_error_series = self._map_to_empirical_distribution(
            ar_series, bin_by_lookahead, self.errors_by_true_value_bin
        )

        return denormalized_error_series

    def _load_errors_by_forecast_bin(self) -> dict[(int, int), np.ndarray]:
        """
        Load the sorted histogram of forecast errors from a pickle file.

        Returns:
            dict[(int, int), np.ndarray]: A dictionary where the keys are lead times and the values are sorted forecast errors.
        """
        with open(
            self.path_data_ar_model_params / f"{self.param}_hist_errors_by_forecast_bin.pickle",
            "rb",
        ) as g:
            errors_by_forecast_bin = pickle.load(g)
        return errors_by_forecast_bin

    def _load_errors_by_true_value_bin(self) -> dict[(int, int), np.ndarray]:
        """
        Load the sorted histogram of forecast errors binned by true value from a pickle file.

        Returns:
            dict[(int, int), np.ndarray]: A dictionary where the keys are lead times and the values are sorted forecast errors.
        """
        with open(
            self.path_data_ar_model_params / f"{self.param}_hist_errors_by_true_value_bin.pickle",
            "rb",
        ) as g:
            errors_by_true_value_bin = pickle.load(g)
        return errors_by_true_value_bin

    def _load_bin_edges(self) -> np.ndarray:
        """
        Load the bin edges from a pickle file.

        Returns:
            np.ndarray: The bin edges.
        """
        with open(self.path_data_ar_model_params / f"{self.param}_bin_edges.pickle", "rb") as f:
            bin_edges = pickle.load(f)
        return bin_edges

    def _load_ar_params(self) -> list[float]:
        """
        Load the AR(2) model parameters from a pickle file.

        Returns:
            list[float]: A list containing the AR(2) model parameters: c, phi_1, phi_2, sigma2.
        """
        with open(self.path_data_ar_model_params / f"{self.param}_ar_params.pickle", "rb") as f:
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
                ar_series[lead_time] = (
                    c + phi_1 * ar_series[lead_time - 1] + np.random.normal(0, sigma)
                )
            else:
                ar_series[lead_time] = (
                    c + phi_1 * ar_series[lead_time - 1] + phi_2 * ar_series[lead_time - 2]
                ) + np.random.normal(0, sigma)

        return ar_series

    def _map_to_empirical_distribution(
        self,
        ar_series: np.ndarray[float],
        bin_by_lookahead: np.ndarray[int],
        errors_by_bin: dict[(int, int), np.ndarray],
    ) -> np.ndarray:
        """
        Map the AR(2) error series to the empirical distribution.

        Args:
            ar_series (np.ndarray): The AR(2) error series.
            bin_by_lookahead (np.ndarray): The bin corresponding to the forecast values.
            errors_by_bin (dict[(int, int), np.ndarray]): The sorted historical forecast errors for each bin and lead time.
        Returns:
            np.ndarray: The denormalized error series.
        """
        forecast_horizon = len(ar_series)
        norm_cdf_percentile_series = norm.cdf(ar_series, loc=0, scale=1)

        # Map the errors based on the percentiles in errors_by_bin
        denormalized_error_series = np.zeros(forecast_horizon)
        for lead_time in range(forecast_horizon):
            bin = bin_by_lookahead[lead_time]
            # Retrieve sorted historical forecast error values
            if len(errors_by_bin[lead_time, bin]) != 0:
                sorted_hist_forecast_error = errors_by_bin[lead_time, bin]
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
