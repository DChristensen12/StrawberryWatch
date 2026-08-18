import matplotlib

# MUST be called before importing pyplot to run without a display
matplotlib.use("Agg")
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from strawberrywatch import paths


def plot_static_dashboard(
    timestamps,
    system_anomaly_scores,
    normalized_anomaly_scores,
    adjusted_thresholds,
    base_threshold,
    spill_flags,
    rain_flags,
    df_original,
    locations,
    threshold_percentile,
):
    """Save the high-resolution Matplotlib dashboard to reports/."""
    spills_during_rain = spill_flags & rain_flags
    spills_no_rain = spill_flags & ~rain_flags

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        timestamps,
        system_anomaly_scores,
        linewidth=1,
        color="blue",
        alpha=0.7,
        label="Anomaly score",
    )
    ax1.plot(
        timestamps,
        adjusted_thresholds,
        color="darkorange",
        linestyle="--",
        linewidth=2.5,
        label="Adjusted threshold",
    )
    ax1.axhline(
        base_threshold,
        color="gold",
        linestyle=":",
        linewidth=2,
        label=f"Base ({threshold_percentile}th %ile)",
    )

    for ts, is_rain in zip(timestamps, rain_flags, strict=True):
        if is_rain:
            ax1.axvspan(ts, ts + pd.Timedelta(minutes=15), alpha=0.1, color="cyan")

    ax1.scatter(
        [timestamps[i] for i in range(len(timestamps)) if spills_during_rain[i]],
        [system_anomaly_scores[i] for i in range(len(timestamps)) if spills_during_rain[i]],
        color="purple",
        s=50,
        zorder=5,
        label="Spill (rain)",
        marker="^",
    )

    ax1.scatter(
        [timestamps[i] for i in range(len(timestamps)) if spills_no_rain[i]],
        [system_anomaly_scores[i] for i in range(len(timestamps)) if spills_no_rain[i]],
        color="red",
        s=50,
        zorder=5,
        label="Spill (no rain)",
        marker="o",
    )

    ax1.set_title("Rain-Aware Spill Detection", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, :])
    mask = (
        (df_original.index >= timestamps[0])
        & (df_original.index <= timestamps[-1])
        & (df_original["location"] == locations[0])
    )
    rain_ts = df_original.loc[mask]["rain_mm"]
    ax2.bar(rain_ts.index, rain_ts.values, width=0.01, color="cyan", alpha=0.7)
    ax2.set_ylabel("Rain (mm)", fontweight="bold")

    ax3 = fig.add_subplot(gs[2, 0])
    sensor_contributions = [
        normalized_anomaly_scores[spill_flags, loc_idx, 0].mean() if np.any(spill_flags) else 0
        for loc_idx in range(len(locations))
    ]
    ax3.barh(locations, sensor_contributions, color="orange")
    ax3.set_title("Sensor Contributions", fontsize=12, fontweight="bold")

    ax4 = fig.add_subplot(gs[2, 1])
    for location in locations:
        loc_data = df_original[df_original["location"] == location]
        ax4.plot(loc_data.index, loc_data["conductivity"], linewidth=1, alpha=0.6, label=location)
    ax4.set_title("Raw Conductivity Data", fontsize=12, fontweight="bold")

    plt.suptitle("SCMG System Results", fontsize=16, fontweight="bold", y=0.995)

    # Created here rather than at import, so bringing in this module never
    # writes to whatever directory the caller happened to be in.
    report_dir = paths.reports_dir()
    os.makedirs(report_dir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save a history file and a 'latest' file for easy checking
    plt.savefig(
        os.path.join(report_dir, f"report_{timestamp_str}.png"), bbox_inches="tight", dpi=150
    )
    plt.savefig(os.path.join(report_dir, "latest_report.png"), bbox_inches="tight", dpi=150)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dashboard saved to {report_dir}/")

    plt.close("all")


def plot_interactive_plotly(
    timestamps,
    system_anomaly_scores,
    adjusted_thresholds,
    base_threshold,
    spill_flags,
    rain_flags,
    rain_threshold_multiplier,
    rain_window_hours,
    threshold_percentile,
):
    """
    Save the interactive Plotly visualization as HTML. plotly.show() won't work
    in a headless loop, which is why we write to disk instead.
    """
    fig_plotly = go.Figure()

    fig_plotly.add_trace(
        go.Scatter(
            x=timestamps,
            y=system_anomaly_scores,
            name="Anomaly Score",
            line=dict(color="royalblue"),
        )
    )
    fig_plotly.add_trace(
        go.Scatter(
            x=timestamps,
            y=adjusted_thresholds,
            name="Adjusted Threshold",
            line=dict(color="darkorange", dash="dash"),
        )
    )

    spills_during_rain = spill_flags & rain_flags

    fig_plotly.add_trace(
        go.Scatter(
            x=[timestamps[i] for i in range(len(timestamps)) if spills_during_rain[i]],
            y=[system_anomaly_scores[i] for i in range(len(timestamps)) if spills_during_rain[i]],
            mode="markers",
            name="Rain Spill",
            marker=dict(color="purple", symbol="triangle-up"),
        )
    )

    fig_plotly.update_layout(
        title=f"Interactive Spill Detection (Multiplier={rain_threshold_multiplier}x)",
        template="plotly_white",
        height=600,
    )

    report_dir = paths.reports_dir()
    os.makedirs(report_dir, exist_ok=True)

    fig_plotly.write_html(os.path.join(report_dir, "interactive_dashboard.html"))
