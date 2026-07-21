"""Day ahead spot-market price forecasting model"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pvlib

from ents_casting.config import config
from ents_casting.forecast_models.lightgbm_model import LgbmForecastModel


def main(skip_hyper_parameter_tuning: bool = False, n_trials: int = 10) -> None:
    if "price_el" not in config["time_series_parameters"]:
        print("Skipping model training of price_el.")
        return
    training_year = config["training_years"]["price_el"]
    model = PriceElModel()

    if not skip_hyper_parameter_tuning:
        print("Tuning lightgbm hyperparameters for price_el...")
        model.tune_hyperparameters(year=training_year, n_trials=n_trials)
    print("Training lightgbm models for price_el...")
    model.cross_validate_and_save_models(year=training_year)


class PriceElModel(LgbmForecastModel):
    def __init__(self) -> None:
        self.ts_param = "price_el"
        self.input_parameters = [
            "co2_price",
            "gas_price",
            "apparent_zenith",
            "azimuth",
            "weekday",
            "hour_sin",
            "hour_cos",
            "lag_mean_24",
            "lag_mean_168",
        ]

        # Add weather data from onshore and offshore locations
        onshore_params = config["onshore_params"]
        offshore_params = config["offshore_params"]
        onshore_locations = np.arange(len(config["onshore_locations"]))
        offshore_locations = np.arange(len(config["offshore_locations"]))

        for loc in onshore_locations:
            for param in onshore_params:
                self.input_parameters.append(f"onshore_{loc}_{param}")

        for loc in offshore_locations:
            for param in offshore_params:
                self.input_parameters.append(f"offshore_{loc}_{param}")

    def load_data(self, year: int) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Loads the data for training or testing the price_el model.

        Args:
            year (int): Year of the measured data to load.
        Returns:
            tuple[np.ndarray, Dict[str, np.ndarray]]: Tuple containing the target values (Y_data) and input features (X_data).
        """
        cwd = Path.cwd()
        complete_data = pd.read_csv(cwd / config["path_data_measured"] / f"data_{year}.csv")
        Y_data = complete_data["price_el"][0:8760].astype(float)

        main_location_data = pd.read_csv(cwd / config["path_data_meteo"] / "main.csv")
        main_location_data = main_location_data[
            (main_location_data["datetime_local"] >= f"{year}-01-01")
            & (main_location_data["datetime_local"] < f"{year + 1}-01-01")
        ][0:8760]
        main_location_data["datetime_local"] = pd.to_datetime(main_location_data["datetime_local"])

        X_data = {
            "co2_price": complete_data["co2_price"].values,
            "gas_price": complete_data["gas_price"].values,
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

        # Add lag features
        for lag in [24, 168]:
            X_data[f"lag_mean_{lag}"] = Y_data.rolling(window=lag).mean().values

        # Add DE weather data
        onshore_locations = np.arange(len(config["onshore_locations"]))
        onshore_params = config["onshore_params"]
        for loc in onshore_locations:
            onshore_data = pd.read_csv(cwd / config["path_data_meteo"] / f"onshore_{loc}.csv")
            # Filter for data in year
            onshore_data = onshore_data[
                (onshore_data["datetime_local"] >= f"{year}-01-01")
                & (onshore_data["datetime_local"] < f"{year + 1}-01-01")
            ][0:8760]
            for param in onshore_params:
                X_data[f"onshore_{loc}_{param}"] = onshore_data[param].values

        offshore_locations = np.arange(len(config["offshore_locations"]))
        offshore_params = config["offshore_params"]
        for loc in offshore_locations:
            offshore_data = pd.read_csv(cwd / config["path_data_meteo"] / f"offshore_{loc}.csv")
            # Filter for data in year
            offshore_data = offshore_data[
                (offshore_data["datetime_local"] >= f"{year}-01-01")
                & (offshore_data["datetime_local"] < f"{year + 1}-01-01")
            ][0:8760]
            for param in offshore_params:
                X_data[f"offshore_{loc}_{param}"] = offshore_data[param].values

        # Convert all data to float
        for key in X_data:
            X_data[key] = X_data[key].astype(float)

        return Y_data, X_data


if __name__ == "__main__":
    main()
