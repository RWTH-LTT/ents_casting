"""Wind turbine model (GE 5.3 158) to calculate the capacity factor based on wind wind_speed data"""

from typing import Dict
import numpy as np


class WindLocalModel:
    """Wind turbine model to calculate the capacity factor based on wind wind_speed data."""

    input_parameters = ["wind_speed_100m", "temperature_2m"]

    def __init__(self) -> None:
        # Reference values for air density calculation
        self.p_ref = 101300  # Pa standard air pressure
        self.t_ref = 15 + 273.15  # K standard temperature
        self.hubheight = 100  # m hub height of the wind turbine

        # Current wind turbine: GE 5.3 158
        self.rated_power = 5300.0  # kW Rated Power of the turbine
        self.cut_in_speed = 3.0  # m/s wind_speed at which the turbine starts producing electricity
        self.rated_speed = 13.0  # m/s wind_speed at which the rated power is reached
        self.cut_out_speed = 22.0  # m/s wind_speed at which the turbine is turned off

        wind_speed_values = np.array(
            [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5]
        )
        power_values = np.array(
            [
                88,
                186,
                310,
                466,
                657,
                892,
                1168,
                1496,
                1876,
                2303,
                2761,
                3227,
                3668,
                4075,
                4452,
                4762,
                4998,
                5165,
                5253,
                5298,
            ]
        )
        self.poly_coeff = np.polyfit(x=wind_speed_values, y=power_values, deg=6)

    def predict(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run the wind turbine model to calculate the capacity factor based on wind wind_speed data.

        Args:
            data (Dict[str, np.ndarray]): Dictionary of 1-D arrays
                'wind_speed_100m' in km/h
                'temperature_2m' in °C.

        Returns:
            np.ndarray: 1-D array of capacity factors for each time step.
        """
        # Check if required input parameters are present
        for param in self.input_parameters:
            if param not in data:
                raise ValueError(f"Missing required input parameter: {param}")

        # Convert km/h to m/s
        wind = data["wind_speed_100m"] / 3.6
        Tair = data["temperature_2m"]

        # Vectorized calculation
        capacity_factor = np.zeros_like(wind)

        # Conditions for partial power
        mask_partial = (self.cut_in_speed <= wind) & (wind <= self.rated_speed)
        if np.any(mask_partial):
            poly_func = np.poly1d(self.poly_coeff)
            capacity_factor[mask_partial] = poly_func(wind[mask_partial]) / self.rated_power

        # Conditions for rated power
        mask_rated = (self.rated_speed <= wind) & (wind <= self.cut_out_speed)
        capacity_factor[mask_rated] = 1.0

        # Temperature correction
        temp_correction = (self.t_ref / (273.15 + Tair - 0.0065 * (self.hubheight - 2))) * np.exp(
            -self.hubheight / 8430
        )
        capacity_factor *= temp_correction

        # Clip to [0, 1]
        capacity_factor = np.clip(capacity_factor, 0, 1)

        return capacity_factor
