"""Download historical weather data from open-meteo.com API."""

from pathlib import Path
import pandas as pd
import openmeteo_requests
import requests_cache
import retry_requests
from datetime import timedelta, timezone

from ents_casting.config import config


def main() -> None:
    download_main_location_historical_data()
    download_onshore_locations_historical_data()
    download_offshore_locations_historical_data()


def download_main_location_historical_data() -> None:
    print("Downloading historical weather data for main location...")
    lat, lon = config["main_location"].split(" ")

    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": f"{config['historical_weather_start_year']}-01-01",
        "end_date": f"{config['historical_weather_end_year']}-12-31",
        "hourly": config["main_params"],
        "timezone": config["timezone"],
    }
    historical_meteo_data = request_from_openmeteo(params)

    # Save the raw historical forecasts to a CSV file
    file_name = Path.cwd() / config["path_data_meteo"] / "main.csv"
    historical_meteo_data.to_csv(file_name)
    print(f"Saved historical weather data for main location in {file_name}.")


def download_onshore_locations_historical_data() -> None:
    print("Downloading historical weather data for onshore locations...")
    for i, loc in enumerate(config["onshore_locations"]):
        lat, lon = loc.split(" ")
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "start_date": f"{config['historical_weather_start_year']}-01-01",
            "end_date": f"{config['historical_weather_end_year']}-12-31",
            "hourly": config["onshore_params"],
            "timezone": config["timezone"],
        }
        historical_meteo_data = request_from_openmeteo(params)

        # Save the raw historical forecasts to a CSV file
        file_name = Path.cwd() / config["path_data_meteo"] / f"onshore_{i}.csv"
        historical_meteo_data.to_csv(file_name)
        print(f"Saved historical weather data for onshore location {i} in {file_name}.")


def download_offshore_locations_historical_data() -> None:
    print("Downloading historical weather data for offshore locations...")
    for i, loc in enumerate(config["offshore_locations"]):
        lat, lon = loc.split(" ")
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "start_date": f"{config['historical_weather_start_year']}-01-01",
            "end_date": f"{config['historical_weather_end_year']}-12-31",
            "hourly": config["offshore_params"],
            "timezone": config["timezone"],
        }
        historical_meteo_data = request_from_openmeteo(params)

        # Save the raw historical forecasts to a CSV file
        file_name = Path.cwd() / config["path_data_meteo"] / f"offshore_{i}.csv"
        historical_meteo_data.to_csv(file_name)
        print(f"Saved historical weather data for offshore location {i} in {file_name}.")


def request_from_openmeteo(params) -> pd.DataFrame:
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry_requests.retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://archive-api.open-meteo.com/v1/archive"

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_data = {
        "datetime_local": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
        )
    }
    hourly_data["datetime_local"] = hourly_data["datetime_local"][:-1]

    for i, var in enumerate(params["hourly"]):
        hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()

    historical_meteo_data = pd.DataFrame(data=hourly_data)
    historical_meteo_data["datetime_local"] = pd.to_datetime(historical_meteo_data["datetime_local"])
    historical_meteo_data.set_index("datetime_local", inplace=True)

    # Enforce non-negativity on some columns
    for column in historical_meteo_data.columns:
        if column in [
            "datetime_local",
            "temperature_2m",
        ]:
            continue
        historical_meteo_data[column] = historical_meteo_data[column].clip(lower=0)

    # Shift solar parameters by one hour because they values from open-meteo are given as the
    # average of the previous hour
    if "direct_normal_irradiance" in historical_meteo_data.columns:
        historical_meteo_data["direct_normal_irradiance"] = (
            historical_meteo_data["direct_normal_irradiance"].shift(-1).fillna(0)
        )
    if "diffuse_radiation" in historical_meteo_data.columns:
        historical_meteo_data["diffuse_radiation"] = historical_meteo_data["diffuse_radiation"].shift(-1).fillna(0)

    # The time index is in UTC timezone. Convert to UTC+1
    tz_offset = int(config["timezone"].replace("UTC", ""))
    tz = timezone(timedelta(hours=tz_offset))  # UTC+1
    historical_meteo_data.index = historical_meteo_data.index.tz_convert(tz)

    return historical_meteo_data


if __name__ == "__main__":
    main()
