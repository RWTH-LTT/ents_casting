"""Plots the stochastic forecast made for a given day in the long-term scenarios."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ents_casting.config import config

YEAR = 2020
DAY = 10  # day at which the forecast starts
HORIZON = 6  # forecast horizon in days
N_FORECASTS = 5

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


def main(year: int = YEAR, day: int = DAY, horizon: int = HORIZON, n_forecasts: int = N_FORECASTS) -> None:
    """Main function to plot the stochastic forecast."""
    path = Path.cwd() / config["path_data_forecasts"]

    with np.load(path / f"perfect_forecasts_{year}.npz") as w:
        perfect_forecasts = {key: w[key] for key in w.files}

    with np.load(path / f"stochastic_forecasts_{year}_{n_forecasts}.npz") as w:
        stochastic_forecasts = {key: w[key] for key in w.files}

    params = [param for param in config["time_series_parameters"] if param in axes_names]
    n_params = len(params)
    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for axis, param in zip(axes, params):
        for scenario in reversed(range(n_forecasts)):
            time_series = [perfect_forecasts[param][day, 0, 0]]
            time_series.extend(list(stochastic_forecasts[param][day, scenario, 1:]))
            if scenario == 0:
                axis.plot(
                    time_series,
                    label="Deterministic forecast",
                    color=parameter_colors[param],
                    linestyle="--",
                )
            elif scenario == 1:
                axis.plot(
                    time_series,
                    label="Forecast scenarios",
                    color=lighten_color(parameter_colors[param], amount=0.4),
                )
            else:
                axis.plot(
                    time_series,
                    color=lighten_color(parameter_colors[param], amount=0.4),
                )

        axis.plot(
            perfect_forecasts[param][day, 0, :],
            label="True values",
            color="black",
            linewidth=1.2,
        )

        axis.tick_params(axis="x", which="minor", bottom=False, top=False)
        axis.tick_params(axis="y", which="minor", left=False, right=False)
        axis.set_xticks([i * 24 for i in range(horizon + 1)])
        axis.set_xlim(0, horizon * 24)
        axis.axhline(0, color="black", linewidth=0.5, linestyle="--")
        axis.set_ylabel(axes_names[param])

    for axis in axes[n_params:]:
        axis.set_axis_off()

    axes[min(n_params - 1, len(axes) - 1)].set_xlabel("Hour in forecast horizon")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_name = f"stochastic_forecast_{year}_day{day}_h{horizon}_summary.pdf"
    fig.savefig(output_name, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot of stochastic forecast to {output_name}.")


# Create ligher color by mixing with white
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    lighter_color = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    return lighter_color


if __name__ == "__main__":
    main()
