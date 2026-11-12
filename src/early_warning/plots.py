"""Charts for the report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .baseline import Baseline, band, failure_starts


def _mark_failures(ax, status: pd.Series) -> None:
    for i, t in enumerate(failure_starts(status)):
        ax.axvline(t, color="#c0392b", lw=1.2, ls="--", label="Recorded failure" if i == 0 else None)


def plot_health(hi: pd.Series, cusum: pd.Series, alerts: pd.DatetimeIndex, status: pd.Series,
                cusum_h: float, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    a1.plot(hi.index, hi, lw=0.6, color="#34495e")
    a1.set_ylabel("Health index\n(mean |z|, top sensors)")
    a1.scatter(alerts, [hi.max() * 1.02] * len(alerts), marker="v", color="#e67e22", s=30, label="Band alert")
    _mark_failures(a1, status)
    a1.legend(loc="upper left", fontsize=8)

    a2.plot(cusum.index, cusum, lw=0.8, color="#2c7fb8")
    a2.axhline(cusum_h, color="#7f8c8d", ls=":", label="CUSUM threshold")
    a2.set_ylabel("CUSUM drift statistic")
    _mark_failures(a2, status)
    a2.legend(loc="upper left", fontsize=8)

    fig.suptitle("Pump health over time")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sensor(data: pd.DataFrame, sensor: str, baseline: Baseline, k: float,
                status: pd.Series, path: Path) -> None:
    lo, hi = band(baseline, sensor, k)
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.plot(data.index, data[sensor], lw=0.5, color="#34495e", label=sensor)
    ax.axhspan(lo, hi, color="#2ecc71", alpha=0.15, label=f"Expected band (±{k:g} robust SD)")
    ax.axvspan(baseline.start, baseline.end, color="#3498db", alpha=0.08, label="Baseline period")
    _mark_failures(ax, status)
    ax.set_title(f"{sensor}: readings against its expected-behaviour band")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_lead_window(data: pd.DataFrame, sz: pd.DataFrame, status: pd.Series, failure: pd.Timestamp,
                     hours: int, path: Path, top: int = 4) -> None:
    """Zoom into the hours before one failure and show the sensors that moved most."""
    win = slice(failure - pd.Timedelta(hours=hours), failure + pd.Timedelta(hours=6))
    movers = sz.loc[win].abs().mean().nlargest(top).index
    fig, axes = plt.subplots(top, 1, figsize=(13, 2.2 * top), sharex=True)
    for ax, s in zip(axes, movers):
        ax.plot(sz.loc[win].index, sz.loc[win, s], lw=0.8, color="#34495e")
        ax.axhline(0, color="#95a5a6", lw=0.6)
        ax.axvline(failure, color="#c0392b", ls="--", lw=1.2)
        ax.set_ylabel(f"{s}\n(z)", fontsize=8)
    axes[0].set_title(f"Sensors that moved most in the {hours} h before the failure at {failure:%Y-%m-%d %H:%M}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
