"""
Generates long-term scenarios based on the historical weather data, the fitted forecast models and
the residual error models (full year).
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pvlib

from ents_casting.config import config
from ents_casting.forecast_models.demand import DemandModel
from ents_casting.forecast_models.price_el import PriceElModel
from ents_casting.forecast_models.emission_el import EmissionElModel
from ents_casting.forecast_models.pv_local import PVlocalModel
from ents_casting.forecast_models.wind_local import WindLocalModel
from ents_casting.ar_models.long_term_error import ErrorGenerator


def main() -> None:
    generate_long_term_scenarios()
    add_ar_error_to_long_term_scenarios()


def generate_long_term_scenarios():
    # Load all input data for forecast models
    weather_data = load_weather_data()
    weather_data = extrapolate_weather_trends(weather_data)
    additional_data = load_additional_data(weather_data["datetime_local"])
    weather_data_years = weather_data["year"].unique()

    print(f"Generating long-term scenarios for weather years {weather_data_years[0]} to {weather_data_years[-1]}...")

    # Merge data to dict of numpy arrays
    X_data_dict = {col: weather_data[col].values for col in weather_data.columns if col != "datetime_local"}
    for col in additional_data.columns:
        if col == "datetime_local":
            continue
        X_data_dict[col] = additional_data[col].values

    long_term_scenarios = pd.DataFrame({"year": X_data_dict["year"]})
    long_term_scenarios["datetime_local"] = additional_data["datetime_local"].values

    # Predict wind_local
    if "wind_local" in config["time_series_parameters"]:
        model = WindLocalModel()
        long_term_scenarios["wind_local"] = model.predict(
            data=X_data_dict,
        )
    else:
        print("Skipping model prediction of wind_local.")
        long_term_scenarios["wind_local"] = np.zeros(len(long_term_scenarios))

    # Predict remaining time series parameters
    ts_params = ["pv_local", "demand_el", "demand_heat", "demand_cold", "price_el", "emission_el"]

    for param in ts_params:
        if param not in config["time_series_parameters"]:
            print(f"Skipping long-term scenario generation for {param}.")
            long_term_scenarios[param] = np.zeros(len(long_term_scenarios))
            continue

        # Load forecast model and measured data from previous year for lag features
        if param == "pv_local":
            model = PVlocalModel()
        elif param in ["demand_el", "demand_heat", "demand_cold"]:
            model = DemandModel(ts_param=param)
        elif param == "price_el":
            model = PriceElModel()
        elif param == "emission_el":
            model = EmissionElModel()
        else:
            raise ValueError(f"Unknown parameter: {param}")

        # Load training data for prediction model
        training_year = config["training_years"][param]
        Y_meas, _ = model.load_data(training_year)

        # Convert to numpy arrays for easier indexing
        y = np.asarray(Y_meas)[0:8760]

        # Iterate through the year in fold-sized steps and load the corresponding trained model
        future_year_scenario = np.zeros(len(long_term_scenarios))
        n_days_per_fold = config["n_days_per_cross_validation_fold"]
        fold_length = n_days_per_fold * 24
        for fold_idx, prediction_interval_start in enumerate(range(0, 8760, fold_length)):
            prediction_interval_end = min(prediction_interval_start + fold_length, len(y))

            # Load the trained model for the current fold
            model.load_gbm_model(fold=fold_idx)

            # Predict for the current interval
            for i, _ in enumerate(weather_data_years):
                start_index = i * 8760 + prediction_interval_start
                end_index = i * 8760 + prediction_interval_end
                prediction = model.predict_with_lag(
                    {key: X_data_dict[key][start_index:end_index] for key in X_data_dict},
                    lag_values=(
                        list(future_year_scenario[start_index - 168 : start_index])
                        if prediction_interval_start >= 168  # else use data from end of year
                        else list(y[-168:])
                    ),
                )
                future_year_scenario[start_index:end_index] = prediction
        long_term_scenarios[param] = future_year_scenario

    # Save the long-term scenarios without AR error
    path = Path.cwd() / config["path_data_long_term_scenarios"]
    long_term_scenarios.to_csv(path / "long_term_scenarios_no_ar_error.csv", index=False)

    print(f"Saved long-term scenarios without AR error to {path / 'long_term_scenarios_no_ar_error.csv'}.")


def add_ar_error_to_long_term_scenarios() -> pd.DataFrame:
    print("Adding AR error to long-term scenarios...")

    # Load the long-term scenarios without AR error
    path = Path.cwd() / config["path_data_long_term_scenarios"]
    long_term_scenarios = pd.read_csv(path / "long_term_scenarios_no_ar_error.csv")

    weather_data = load_weather_data()
    additional_data = load_additional_data(weather_data["datetime_local"])

    # Predict residual error and post-process each parameter
    ts_params = ["pv_local", "demand_el", "demand_heat", "demand_cold", "price_el", "emission_el"]

    for param in ts_params:
        if param not in long_term_scenarios.columns:
            continue

        ar_model = ErrorGenerator(param=param)
        error_series = ar_model.generate_error_series(forecast=long_term_scenarios[param].values)
        long_term_scenarios[param] += error_series

        if param != "price_el":
            long_term_scenarios[param] = long_term_scenarios[param].clip(lower=0.0)

        if param == "pv_local":
            solar_elevation = additional_data["apparent_zenith"].values
            long_term_scenarios.loc[solar_elevation >= 90, "pv_local"] = 0.0

    long_term_scenarios_to_save = long_term_scenarios.drop(columns=["datetime_local"])

    # Save the long-term scenarios
    path = Path.cwd() / config["path_data_long_term_scenarios"]
    long_term_scenarios_to_save.to_csv(path / "long_term_scenarios.csv", index=False)

    print(f"Saved long-term scenarios to {path / 'long_term_scenarios.csv'}.")


def load_weather_data() -> pd.DataFrame:
    """
    Loads historical weather data from CSV files.

    Returns:
        pd.DataFrame: A DataFrame containing weather parameters as columns and their corresponding data for all years as rows.
    """
    cwd = Path.cwd()

    # Load main location weather data
    weather_data = pd.read_csv(cwd / config["path_data_meteo"] / "main.csv")
    weather_data["datetime_local"] = pd.to_datetime(weather_data["datetime_local"])

    # Load onshore weather data
    onshore_locations = np.arange(len(config["onshore_locations"]))
    onshore_features = config["onshore_params"]
    for loc in onshore_locations:
        onshore_data = pd.read_csv(cwd / config["path_data_meteo"] / f"onshore_{loc}.csv")
        for feature in onshore_features:
            weather_data[f"onshore_{loc}_{feature}"] = onshore_data[feature].values

    # Load offshore weather data
    offshore_locations = np.arange(len(config["offshore_locations"]))
    offshore_features = config["offshore_params"]
    for loc in offshore_locations:
        offshore_data = pd.read_csv(cwd / config["path_data_meteo"] / f"offshore_{loc}.csv")
        for feature in offshore_features:
            weather_data[f"offshore_{loc}_{feature}"] = offshore_data[feature].values

    # Add year and hour in year
    weather_data["year"] = weather_data["datetime_local"].dt.year
    weather_data["hour_in_year"] = (weather_data["datetime_local"].dt.dayofyear - 1) * 24 + weather_data[
        "datetime_local"
    ].dt.hour

    # Cutoff all rows for hour_in_year >= 8760
    weather_data = weather_data[weather_data["hour_in_year"] < 8760].reset_index(drop=True)

    return weather_data


def extrapolate_weather_trends(
    weather_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Linearly extrapolate historical weather data to the year ahead.

    Args:
        weather_data (pd.DataFrame): A DataFrame containing the historical weather data over multiple years.

    Returns:
        pd.DataFrame: A DataFrame containing the historical weather data where each year is extrapolated to the year ahead.
    """
    print("Extrapolating weather trends for the long-term scenarios...")
    # Get the list of all weather parameters that include temperature
    weather_params = []
    main_params = config["main_params"]
    weather_params.extend(main_params)
    onshore_params = config["onshore_params"]
    n_onshore_locations = len(config["onshore_locations"])
    for loc in range(n_onshore_locations):
        for param in onshore_params:
            weather_params.append(f"onshore_{loc}_{param}")
    offshore_params = config["offshore_params"]
    n_offshore_locations = len(config["offshore_locations"])
    for loc in range(n_offshore_locations):
        for param in offshore_params:
            weather_params.append(f"offshore_{loc}_{param}")

    # Get the linear slopes in x/per year for each time series parameter
    slopes = {}
    for param in weather_params:
        # Fit a linear regression model to the historical data
        average_value_by_year = weather_data.groupby("year")[param].mean().reset_index()
        # Fit linear regression: value ≈ slope * year + intercept
        slopes[param], _ = np.polyfit(
            average_value_by_year["year"], average_value_by_year[param], 1  # degree of polynomial
        )

    # Extrapolate each year to the year ahead
    weather_data_extrapolated = weather_data.copy()
    max_year = weather_data["year"].max()
    for year in weather_data["year"].unique():
        for param in weather_params:
            weather_data_extrapolated.loc[weather_data_extrapolated["year"] == year, param] += slopes[param] * (
                max_year + 1 - year
            )

    return weather_data_extrapolated


def load_measured_data(year: int) -> Dict[str, np.ndarray]:
    """
    Loads measured data for a specific year.

    Args:
        year (int): The year of the measured data to load.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing measured parameters as keys and their corresponding data as numpy arrays.
    """
    cwd = Path.cwd()
    measured_data = {}

    # Load measured data for the specified year from the new combined file
    complete_data = pd.read_csv(cwd / config["path_data_measured"] / f"data_{year}.csv")

    for column in complete_data.columns:
        if column != "datetime_local":
            measured_data[column] = complete_data[column].values

    return measured_data


def load_additional_data(time: pd.DataFrame) -> pd.DataFrame:
    """
    Add further input data required for the forecast models unrelated to weather data.

    Args:
        time (pd.DataFrame): A DataFrame containing the time column.

    Returns:
        pd.DataFrame: A DataFrame containing all input parameters as columns and their corresponding data for all years as rows.
    """
    additional_data = pd.DataFrame(time)

    # add year
    additional_data["year"] = additional_data["datetime_local"].dt.year

    # Add solar elevation and azimuth
    lat, lon = config["main_location"].split(" ")
    lat = float(lat)
    lon = float(lon)
    solpos = pvlib.solarposition.get_solarposition(time, lat, lon)
    additional_data["apparent_zenith"] = solpos["apparent_zenith"].values
    additional_data["azimuth"] = solpos["azimuth"].values

    # Add daytime features sin and cos of hour of day
    hours = time.dt.hour.values
    additional_data["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    additional_data["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Add calendar features, treating public holidays as sundays
    datetime_index_2024 = pd.date_range(start="2024-01-01", end="2024-12-30 23:00:00", freq="h")
    public_holidays = config.get("holidays", [])
    public_holidays = pd.to_datetime(public_holidays).normalize()

    is_public_holiday = datetime_index_2024.normalize().isin(public_holidays)

    weekday_df = pd.DataFrame(datetime_index_2024.weekday, columns=["weekday"])

    weekday_df["weekday"] = weekday_df["weekday"].where(~is_public_holiday, 6)

    # reduce length of weekday_df to the length of additional_data
    if len(weekday_df) > len(additional_data):
        weekday_df = weekday_df.iloc[: len(additional_data)]
    weekday_array = weekday_df["weekday"].to_numpy()

    cwd = Path.cwd()
    fuel_price_file_path = cwd / config["path_data_measured"] / f"data_{config['training_years']['price_el']}.csv"
    complete_fuel_price_data = pd.read_csv(fuel_price_file_path)

    n_years = additional_data["year"].nunique()

    additional_data["weekday"] = np.tile(weekday_array, n_years)
    additional_data["co2_price"] = complete_fuel_price_data["co2_price"].mean()
    additional_data["gas_price"] = complete_fuel_price_data["gas_price"].mean()

    return additional_data


if __name__ == "__main__":
    main()
