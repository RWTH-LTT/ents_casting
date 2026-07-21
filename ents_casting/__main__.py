"""Main entry point for ents_casting."""

import ents_casting.meteo_downloaders.historical_weather
import ents_casting.meteo_downloaders.historical_weather_forecasts

import ents_casting.forecast_models.demand
import ents_casting.forecast_models.pv_local
import ents_casting.forecast_models.price_el
import ents_casting.forecast_models.emission_el

import ents_casting.generate_long_term_scenarios
import ents_casting.residual_error.fit_residual_error_long_term
import ents_casting.plotting.plot_long_term_scenarios

import ents_casting.weather_forecast_error.historical_forecasting_error
import ents_casting.weather_forecast_error.fit_forecast_error
import ents_casting.residual_error.fit_residual_error
import ents_casting.generate_forecasts
import ents_casting.plotting.plot_stochastic_forecast
import ents_casting.plotting.plot_forecast_accuracy

# Step 1: Download the weather data for the specified weather years and locations
ents_casting.meteo_downloaders.historical_weather.main()

# Step 2: Download the historical weather forecasts
ents_casting.meteo_downloaders.historical_weather_forecasts.main()

# Step 3: Train the lightGBM forecast models including k-fold cross-validation
ents_casting.forecast_models.demand.main(skip_hyper_parameter_tuning=False, n_trials=5)
ents_casting.forecast_models.pv_local.main(skip_hyper_parameter_tuning=False, n_trials=5)
ents_casting.forecast_models.price_el.main(skip_hyper_parameter_tuning=False, n_trials=5)
ents_casting.forecast_models.emission_el.main(skip_hyper_parameter_tuning=False, n_trials=5)

# Step 4: Generate the long-term scenarios for the specified weather years using the trained lightGBM models
ents_casting.generate_long_term_scenarios.generate_long_term_scenarios()
ents_casting.plotting.plot_long_term_scenarios.main(with_ar_error=False)

# Step 5: Fit AR models to the error between the generated long-term scenarios and the measured data.
ents_casting.residual_error.fit_residual_error_long_term.main()

# Step 6: Add AR noise to the long-term scenarios based on the fitted AR models for the residuals
ents_casting.generate_long_term_scenarios.add_ar_error_to_long_term_scenarios()
ents_casting.plotting.plot_long_term_scenarios.main(with_ar_error=True)

# Step 7: Train the AR Error models for weather forecasts
ents_casting.weather_forecast_error.historical_forecasting_error.main()
ents_casting.weather_forecast_error.fit_forecast_error.main()

# Step 8: Train the AR Error models for residuals between the lightGBM forecasts and measured data
ents_casting.residual_error.fit_residual_error.main()

# Step 9: Generate the short-term forecasts for each day of the long-term scenarios
ents_casting.generate_forecasts.main()

# Plot an exemplary stochastic forecast for a specific day and horizon
ents_casting.plotting.plot_stochastic_forecast.main(year=2024, day=10, n_forecasts=5)

# Plot the accuracy statistics of the short-term forecasts over the forecast horizon
ents_casting.plotting.plot_forecast_accuracy.main(year=2024, n_forecasts=5)
