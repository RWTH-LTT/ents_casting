"""
Demand models for heating, cooling, and electric demand based on weather parameters and calendar
information.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pvlib

from ents_casting.config import config
from ents_casting.forecast_models.lightgbm_model import LgbmForecastModel


def main(skip_hyper_parameter_tuning: bool = False, n_trials: int = 10) -> None:
    """Train the demand models for electricity, heating, and cooling."""
    for ts_param in ["demand_el", "demand_heat", "demand_cold"]:
        if ts_param not in config["time_series_parameters"]:
            print(f"Skipping model training of {ts_param}.")
            continue

        training_year = config["training_years"][ts_param]
        model = DemandModel(ts_param=ts_param)

        if not skip_hyper_parameter_tuning:
            print(f"Tuning lightgbm hyperparameters for {ts_param}...")
            model.tune_hyperparameters(year=training_year, n_trials=n_trials)
        print(f"Training lightgbm models for {ts_param}...")
        model.cross_validate_and_save_models(year=training_year)


class DemandModel(LgbmForecastModel):
    def __init__(self, ts_param: str) -> None:
        self.ts_param = ts_param
        self.input_parameters = [
            "temperature_2m",
            "direct_normal_irradiance",
            "shortwave_radiation",
            "wind_speed_10m",
            "apparent_zenith",
            "azimuth",
            "weekday",
            "hour_sin",
            "hour_cos",
            "lag_mean_24",
            "lag_mean_168",
        ]

    def load_data(self, year: int) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Loads the data for training or testing the demand models.

        Args:
            year (int): Year of the measured data to load.

        Returns:
            tuple[np.ndarray, Dict[str, np.ndarray]]: Tuple containing the target values (Y_data) and input features (X_data).
        """
        cwd = Path.cwd()
        complete_data = pd.read_csv(cwd / config["path_data_measured"] / f"data_{year}.csv")
        Y_data = complete_data[self.ts_param][0:8760].astype(float)

        # Load local weather data
        main_location_data = pd.read_csv(cwd / config["path_data_meteo"] / "main.csv")
        main_location_data = main_location_data[
            (main_location_data["datetime_local"] >= f"{year}-01-01")
            & (main_location_data["datetime_local"] < f"{year + 1}-01-01")
        ][0:8760]

        X_data = {
            "temperature_2m": main_location_data["temperature_2m"].values,
            "direct_normal_irradiance": main_location_data["direct_normal_irradiance"].values,
            "shortwave_radiation": main_location_data["shortwave_radiation"].values,
            "wind_speed_10m": main_location_data["wind_speed_10m"].values,
        }

        # Add solar elevation and azimuth
        lat, lon = config["main_location"].split(" ")
        lat = float(lat)
        lon = float(lon)
        main_location_data["datetime_local"] = pd.to_datetime(main_location_data["datetime_local"])
        solpos = pvlib.solarposition.get_solarposition(main_location_data["datetime_local"], lat, lon)
        X_data["apparent_zenith"] = solpos["apparent_zenith"].values
        X_data["azimuth"] = solpos["azimuth"].values

        # Add calendar features and treat public holidays as Sundays (Sunday=6)
        public_holidays = config.get("holidays", [])
        public_holidays = pd.to_datetime(public_holidays)
        is_public_holiday = main_location_data["datetime_local"].dt.date.isin(public_holidays.date)
        X_data["weekday"] = main_location_data["datetime_local"].dt.weekday
        X_data["weekday"] = X_data["weekday"].where(~is_public_holiday, 6).to_numpy()

        # Add daytime features sin and cos of hour of day
        hours = main_location_data["datetime_local"].dt.hour.values
        X_data["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        X_data["hour_cos"] = np.cos(2 * np.pi * hours / 24)

        # Convert all data to float
        for key in X_data:
            X_data[key] = X_data[key].astype(float)

        # Add lag features
        for lag in [24, 168]:
            X_data[f"lag_mean_{lag}"] = Y_data.rolling(window=lag).mean().values

        return Y_data, X_data


if __name__ == "__main__":
    main()
