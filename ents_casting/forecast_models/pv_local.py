"""Photovoltaic capacity factor forecasting model."""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pvlib

from ents_casting.config import config
from ents_casting.forecast_models.lightgbm_model import LgbmForecastModel


def main(skip_hyper_parameter_tuning: bool = False, n_trials: int = 10) -> None:
    if "pv_local" not in config["time_series_parameters"]:
        print("Skipping model training of pv_local.")
        return
    training_year = config["training_years"]["pv_local"]
    model = PVlocalModel()

    if not skip_hyper_parameter_tuning:
        print("Tuning lightgbm hyperparameters for pv_local...")
        model.tune_hyperparameters(year=training_year, n_trials=n_trials)
    print("Training lightgbm models for pv_local...")
    model.cross_validate_and_save_models(year=training_year)


class PVlocalModel(LgbmForecastModel):
    def __init__(self) -> None:
        self.ts_param = "pv_local"
        self.input_parameters = [
            "temperature_2m",
            "direct_normal_irradiance",
            "shortwave_radiation",
            "wind_speed_10m",
            "apparent_zenith",
            "azimuth",
            "lag_mean_24",
            "lag_mean_168",
        ]

    def load_data(self, year: int) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Loads the data for training or testing the PV local model.

        Args:
            year (int): Year of the measured data to load.
        Returns:
            tuple[np.ndarray, Dict[str, np.ndarray]]: Tuple containing the target values (Y_data) and input features (X_data).
        """
        cwd = Path.cwd()
        complete_data = pd.read_csv(cwd / config["path_data_measured"] / f"data_{year}.csv")
        Y_data = complete_data["pv_local"][0:8760].astype(float)

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

        # Convert all data to float
        for key in X_data:
            X_data[key] = X_data[key].astype(float)

        # Add lag features
        for lag in [24, 168]:
            X_data[f"lag_mean_{lag}"] = Y_data.rolling(window=lag).mean().values

        return Y_data, X_data


if __name__ == "__main__":
    main()
