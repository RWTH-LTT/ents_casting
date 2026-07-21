"""Plots the accuracy of the short-term stochastic forecast."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from ents_casting.config import config

YEAR = 2020
N_FORECASTS = 5

parameters = config["time_series_parameters"]
axes_names = {
    "demand_heat": "Heat demand\nin MW",
    "demand_cold": "Cold demand\nin MW",
    "demand_el": "Electricity demand\nin MW",
    "wind_local": "Wind capacity\nfactor",
    "pv_local": "Photovoltaic\ncapacity factor",
    "price_el": "Electricity price\nin €/MWh",
    "emission_el": "Emission factor\nin kg/MWh",
}


parameter_colors = {
    "demand_heat": "#C83E4D",  # muted red
    "demand_cold": "#2CB7B0",  # turquoise
    "demand_el": "#80839B",  # grey
    "wind_local": "#1E78C2",  # blue
    "pv_local": "#FFE000",  # yellow
    "price_el": "#7B3294",  # purple
    "emission_el": "#EA5297",  # pink
}


def main(year: int = YEAR, n_forecasts: int = N_FORECASTS) -> None:
    """Backward-compatible wrapper that creates both forecast-accuracy plots."""
    plot_hourly_MAE_over_forecast_horizon(year, n_forecasts)
    plot_daily_MAE_over_forecast_horizon(year, n_forecasts)


def _load_forecast_data(year: int, n_forecasts: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = Path.cwd() / config["path_data_forecasts"]
    with np.load(path / f"perfect_forecasts_{year}.npz") as w:
        perfect_forecasts = {key: w[key] for key in w.files}

    with np.load(path / f"stochastic_forecasts_{year}_{n_forecasts}.npz") as w:
        stochastic_forecasts = {key: w[key] for key in w.files}

    return perfect_forecasts, stochastic_forecasts


def plot_hourly_MAE_over_forecast_horizon(year: int = YEAR, n_forecasts: int = N_FORECASTS) -> None:
    """Plot mean absolute error over individual forecast lead-time hours."""
    perfect_forecasts, stochastic_forecasts = _load_forecast_data(year, n_forecasts)

    n_rows = int(np.ceil(len(parameters) / 2))
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    print(f"Residual error metrics (short-term forecasts with imperfect weather forecasts):")
    for parameter_idx, parameter in enumerate(parameters):
        ax = axes[parameter_idx]

        synthetic_errors = stochastic_forecasts[parameter][:, 0, :] - perfect_forecasts[parameter][:, 0, :]
        forecast_horizon = synthetic_errors.shape[-1]
        errors_by_sample = synthetic_errors.reshape(-1, forecast_horizon)
        absolute_errors = np.abs(errors_by_sample)

        mae_by_leadtime = np.mean(absolute_errors, axis=0)
        lower_90 = np.percentile(absolute_errors, 2.5, axis=0)
        lower_50 = np.percentile(absolute_errors, 25, axis=0)
        upper_50 = np.percentile(absolute_errors, 75, axis=0)
        upper_90 = np.percentile(absolute_errors, 97.5, axis=0)

        x_ticks_hours = np.arange(0, forecast_horizon + 1, 24)
        ax.set_xlim(0, forecast_horizon)
        ax.set_xticks(x_ticks_hours)
        ax.tick_params(
            axis="x", which="major", top=True, bottom=True, labelbottom=(parameter_idx >= len(parameters) - 2)
        )
        ax.tick_params(axis="x", which="minor", top=False, bottom=False)
        ax.tick_params(axis="y", which="major", left=True, right=True, labelright=False)
        ax.tick_params(axis="y", which="minor", left=False, right=False)
        ax.set_ylabel(axes_names[parameter])
        if parameter_idx >= len(parameters) - 2:
            ax.set_xlabel("Hour in forecast horizon")

        color = parameter_colors[parameter]
        ax.fill_between(range(forecast_horizon), lower_90, upper_90, color=color, alpha=0.12)
        ax.fill_between(range(forecast_horizon), lower_50, upper_50, color=color, alpha=0.23)
        ax.plot(range(forecast_horizon), mae_by_leadtime, color=color)
        ax.grid(False)

        print(f"MAE of {parameter}: {mae_by_leadtime.mean():.4f}")

    for extra_idx in range(len(parameters), n_rows * 2):
        axes[extra_idx].set_axis_off()

    axes[0].set_title("Hourly mean absolute error")
    axes[1].set_title("Hourly mean absolute error")

    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_name = f"forecast_hourly_accuracy_over_horizon_{year}.pdf"
    fig.savefig(output_name, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved hourly forecast accuracy plot to {output_name}.")


def plot_daily_MAE_over_forecast_horizon(year: int = YEAR, n_forecasts: int = N_FORECASTS) -> None:
    """Plot mean absolute error of daily-averaged forecast values."""
    perfect_forecasts, stochastic_forecasts = _load_forecast_data(year, n_forecasts)

    n_rows = int(np.ceil(len(parameters) / 2))
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for parameter_idx, parameter in enumerate(parameters):
        ax = axes[parameter_idx]

        synthetic_errors = stochastic_forecasts[parameter][:, 0, :] - perfect_forecasts[parameter][:, 0, :]
        forecast_horizon = synthetic_errors.shape[-1]
        errors_by_sample = synthetic_errors.reshape(-1, forecast_horizon)

        day_starts = list(range(0, forecast_horizon, 24))
        n_days = len(day_starts)
        daily_abs_mean_errors = np.empty((errors_by_sample.shape[0], n_days), dtype=float)
        for day_idx, start in enumerate(day_starts):
            end = min(start + 24, forecast_horizon)
            daily_abs_mean_errors[:, day_idx] = np.abs(np.mean(errors_by_sample[:, start:end], axis=1))

        daily_mae = np.mean(daily_abs_mean_errors, axis=0)
        daily_lower_90 = np.percentile(daily_abs_mean_errors, 2.5, axis=0)
        daily_lower_50 = np.percentile(daily_abs_mean_errors, 25, axis=0)
        daily_upper_50 = np.percentile(daily_abs_mean_errors, 75, axis=0)
        daily_upper_90 = np.percentile(daily_abs_mean_errors, 97.5, axis=0)

        x_edges = np.arange(0, n_days + 1)
        y_daily = np.r_[daily_mae, daily_mae[-1]]
        y_daily_l90 = np.r_[daily_lower_90, daily_lower_90[-1]]
        y_daily_u90 = np.r_[daily_upper_90, daily_upper_90[-1]]
        y_daily_l50 = np.r_[daily_lower_50, daily_lower_50[-1]]
        y_daily_u50 = np.r_[daily_upper_50, daily_upper_50[-1]]

        ax.set_xlim(0, n_days)
        ax.tick_params(
            axis="x", which="major", top=True, bottom=True, labelbottom=(parameter_idx >= len(parameters) - 2)
        )
        ax.tick_params(axis="x", which="minor", top=False, bottom=False)
        ax.tick_params(axis="y", which="major", left=True, right=True, labelright=False)
        ax.tick_params(axis="y", which="minor", left=False, right=False)
        ax.set_ylabel(axes_names[parameter])
        if parameter_idx >= len(parameters) - 2:
            ax.set_xlabel("Day in forecast horizon")

        color = parameter_colors[parameter]
        ax.fill_between(x_edges, y_daily_l90, y_daily_u90, color=color, alpha=0.12, step="post")
        ax.fill_between(x_edges, y_daily_l50, y_daily_u50, color=color, alpha=0.23, step="post")
        ax.step(x_edges, y_daily, color=color, where="post")
        ax.grid(False)

        print(f"MAE of daily mean {parameter}: {daily_mae.mean():.4f}")

    for extra_idx in range(len(parameters), n_rows * 2):
        axes[extra_idx].set_axis_off()

    axes[0].set_title("Mean absolute error of daily averages")
    axes[1].set_title("Mean absolute error of daily averages")

    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_name = f"forecast_daily_accuracy_over_horizon_{year}.pdf"
    fig.savefig(output_name, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved daily forecast accuracy plot to {output_name}.")


if __name__ == "__main__":
    main()
