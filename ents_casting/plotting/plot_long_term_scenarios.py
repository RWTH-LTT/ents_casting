"""Plots the long-term scenarios distribution over all years."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd

from ents_casting.config import config

axes_names = {
    "demand_heat": "Daily mean heat\ndemand in MW",
    "demand_cold": "Daily mean cold\ndemand in MW",
    "demand_el": "Daily mean electricity\ndemand in MW",
    "wind_local": "Daily mean wind\ncapacityfactor",
    "pv_local": "Daily mean photovoltaic\ncapacity factor",
    "price_el": "Daily mean electricity\nprice in €/MWh",
    "emission_el": "Daily mean emission\nfactor in kg/MWh",
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


def main(with_ar_error: bool = True) -> None:
    path = Path.cwd() / config["path_data_long_term_scenarios"]
    file_name = "long_term_scenarios.csv" if with_ar_error else "long_term_scenarios_no_ar_error.csv"
    long_term_scenarios = pd.read_csv(path / file_name)

    params = [param for param in config["time_series_parameters"] if param in axes_names]
    n_params = len(params)
    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))

    full_index = pd.date_range(start="2024-01-01", periods=365, freq="D")
    month_starts = pd.date_range(full_index[0], full_index[-1], freq="MS")
    next_start = month_starts[-1] + pd.offsets.MonthBegin(1)
    last_end = min(next_start, full_index[-1])
    month_ends = list(month_starts[1:]) + [last_end]
    midpoints = [start + (end - start) / 2 for start, end in zip(month_starts, month_ends)]
    month_midpoints_num = mdates.date2num(midpoints)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    representative_year = 2024

    for axis, param in zip(axes, params):
        yearly_daily_profiles = []
        yearly_values = sorted(long_term_scenarios["year"].unique())
        representative_profile = None

        for year in yearly_values:
            yearly_series = long_term_scenarios.loc[long_term_scenarios["year"] == year, param].to_numpy()
            day_count = min(365, yearly_series.size // 24)
            if day_count == 0:
                continue

            daily_profile = yearly_series[: day_count * 24].reshape(day_count, 24).mean(axis=1)
            yearly_daily_profiles.append(daily_profile)

            if year == representative_year:
                representative_profile = daily_profile

        if not yearly_daily_profiles:
            axis.set_axis_off()
            continue

        profiles = np.vstack(yearly_daily_profiles)
        median = np.median(profiles, axis=0)
        lower_50 = np.percentile(profiles, 25, axis=0)
        upper_50 = np.percentile(profiles, 75, axis=0)
        lower_95 = np.percentile(profiles, 2.5, axis=0)
        upper_95 = np.percentile(profiles, 97.5, axis=0)

        if representative_profile is None:
            representative_profile = profiles[-1]

        color = parameter_colors.get(param)

        axis.fill_between(
            full_index[: len(lower_95)],
            lower_95,
            upper_95,
            color=to_rgba(color, 0.12),
            linewidth=0,
            label="95% interval",
        )
        axis.fill_between(
            full_index[: len(lower_50)],
            lower_50,
            upper_50,
            color=to_rgba(color, 0.28),
            linewidth=0,
            label="50% interval",
        )
        axis.plot(full_index[: len(median)], median, color=color, linewidth=1.8, label="Median")
        axis.plot(
            full_index[: len(representative_profile)],
            representative_profile,
            color="black",
            linewidth=1.1,
            linestyle="--",
            label="2024",
        )

        axis.axhline(0, color="black", linewidth=0.5, linestyle="--")
        axis.set_ylabel(axes_names[param])
        axis.set_xlim(full_index[0], full_index[-1])
        axis.tick_params(axis="y", which="major", left=True, right=False, labelright=False)

        axis.set_xticks(month_starts)
        axis.xaxis.set_minor_locator(FixedLocator(month_midpoints_num))
        axis.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
        axis.tick_params(axis="x", which="major", bottom=True, labelbottom=False)
        axis.tick_params(axis="x", which="minor", bottom=True, length=0, width=0, labelbottom=True)

    for axis in axes[n_params:]:
        axis.set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_name = (
        "long_term_scenarios_summary_with_ar_error.pdf"
        if with_ar_error
        else "long_term_scenarios_summary_no_ar_error.pdf"
    )
    fig.savefig(output_name, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved long-term scenarios summary as {output_name}")


if __name__ == "__main__":
    main(with_ar_error=True)
    main(with_ar_error=False)
