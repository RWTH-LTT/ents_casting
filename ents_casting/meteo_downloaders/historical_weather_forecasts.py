"""Download previous weather forecasts from open-meteo.com API."""

from pathlib import Path
from typing import List
import pandas as pd
import openmeteo_requests
import requests_cache
import retry_requests
from datetime import timedelta, timezone

from ents_casting.config import config


def main() -> None:
    download_main_location_forecasts()
    download_onshore_locations_forecasts()
    download_offshore_locations_forecasts()


def download_main_location_forecasts() -> None:
    print("Downloading historical weather forecasts for main location...")
    lat, lon = config["main_location"].split(" ")
    weather_params = config["main_params"]

    forecast_data = request_from_openmeteo(lat, lon, weather_params)

    # Save forecast data to a CSV file
    file_name = Path.cwd() / config["path_data_meteo"] / "forecasts_main.csv"
    forecast_data.to_csv(file_name)
    print(f"Saved historical forecasts for main location in {file_name}.")


def download_onshore_locations_forecasts() -> None:
    print("Downloading historical weather forecasts for onshore locations...")
    weather_params = config["onshore_params"]
    for i, loc in enumerate(config["onshore_locations"]):
        lat, lon = loc.split(" ")
        forecast_data = request_from_openmeteo(lat, lon, weather_params)
        file_name = Path.cwd() / config["path_data_meteo"] / f"forecasts_onshore_{i}.csv"
        forecast_data.to_csv(file_name)
        print(f"Saved historical forecasts for onshore location {i}")


def download_offshore_locations_forecasts() -> None:
    print("Downloading historical weather forecasts for offshore locations...")
    weather_params = config["offshore_params"]
    for i, loc in enumerate(config["offshore_locations"]):
        lat, lon = loc.split(" ")
        forecast_data = request_from_openmeteo(lat, lon, weather_params)
        file_name = Path.cwd() / config["path_data_meteo"] / f"forecasts_offshore_{i}.csv"
        forecast_data.to_csv(file_name)
        print(f"Saved historical forecasts for offshore location {i}")


def request_from_openmeteo(lat: float, lon: float, weather_params: List[str]) -> pd.DataFrame:
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry_requests.retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"

    # Replace wind_speed_100m with wind_speed_80m and wind_speed_120m due to data availability
    extended_weather_params = weather_params.copy()
    if "wind_speed_100m" in extended_weather_params:
        extended_weather_params.remove("wind_speed_100m")
        extended_weather_params.append("wind_speed_80m")
        extended_weather_params.append("wind_speed_120m")

    parameters = []
    for p in extended_weather_params:
        parameters.append(p)
        for leadtime_day in range(1, 7):
            parameters.append(f"{p}_previous_day{leadtime_day}")

    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": parameters,
        "start_date": config["historical_forecast_start_date"],
        "end_date": config["historical_forecast_end_date"],
        "timezone": config["timezone"],
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
        )
    }
    hourly_data["time"] = hourly_data["time"][:-1]

    for i, var in enumerate(params["hourly"]):
        hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()

    historical_meteo_forecasts = pd.DataFrame(data=hourly_data)
    historical_meteo_forecasts["time"] = pd.to_datetime(historical_meteo_forecasts["time"])
    historical_meteo_forecasts.set_index("time", inplace=True)

    # Enforce non-negativity on all but the temperature parameters
    for column in historical_meteo_forecasts.columns:
        if column == "time" or "temperature_2m" in column:
            continue
        historical_meteo_forecasts[column] = historical_meteo_forecasts[column].clip(lower=0)

    # Shift solar parameters by one hour because they values from open-meteo are given as the
    # average of the previous hour
    if "direct_normal_irradiance" in historical_meteo_forecasts.columns:
        historical_meteo_forecasts["direct_normal_irradiance"] = (
            historical_meteo_forecasts["direct_normal_irradiance"].shift(-1).fillna(0)
        )
    if "diffuse_radiation" in historical_meteo_forecasts.columns:
        historical_meteo_forecasts["diffuse_radiation"] = (
            historical_meteo_forecasts["diffuse_radiation"].shift(-1).fillna(0)
        )
    if "shortwave_radiation" in historical_meteo_forecasts.columns:
        historical_meteo_forecasts["shortwave_radiation"] = (
            historical_meteo_forecasts["shortwave_radiation"].shift(-1).fillna(0)
        )

    # Average wind speed to 100m height
    historical_meteo_forecasts["wind_speed_100m"] = (
        historical_meteo_forecasts["wind_speed_80m"] + historical_meteo_forecasts["wind_speed_120m"]
    ) / 2
    historical_meteo_forecasts.drop(columns=["wind_speed_80m", "wind_speed_120m"], inplace=True)
    for day in range(1, 7):
        historical_meteo_forecasts[f"wind_speed_100m_previous_day{day}"] = (
            historical_meteo_forecasts[f"wind_speed_80m_previous_day{day}"]
            + historical_meteo_forecasts[f"wind_speed_120m_previous_day{day}"]
        ) / 2
        historical_meteo_forecasts.drop(
            columns=[f"wind_speed_80m_previous_day{day}", f"wind_speed_120m_previous_day{day}"],
            inplace=True,
        )

    # The time index is in UTC timezone. Convert to UTC+1
    tz_offset = int(config["timezone"].replace("UTC", ""))
    tz = timezone(timedelta(hours=tz_offset))  # UTC+1
    historical_meteo_forecasts.index = historical_meteo_forecasts.index.tz_convert(tz)

    return historical_meteo_forecasts


if __name__ == "__main__":
    main()
