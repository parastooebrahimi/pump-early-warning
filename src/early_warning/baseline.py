"""Learn each sensor's expected behaviour from a healthy period."""

from dataclasses import dataclass

import pandas as pd

from .config import Config


@dataclass
class Baseline:
    start: pd.Timestamp
    end: pd.Timestamp
    median: pd.Series
    scale: pd.Series  # robust standard deviation (1.4826 x MAD)


def failure_starts(status: pd.Series) -> pd.DatetimeIndex:
    """Timestamps where the machine changes into BROKEN."""
    broken = status.eq("BROKEN")
    return status.index[broken & ~broken.shift(fill_value=False)]


def healthy_mask(status: pd.Series, cfg: Config) -> pd.Series:
    """True where the pump is NORMAL and not in the run-up to a failure."""
    mask = status.eq("NORMAL")
    for t in failure_starts(status):
        mask &= ~((status.index >= t - pd.Timedelta(hours=cfg.exclude_before_failure_h)) & (status.index < t))
    return mask


def fit_baseline(data: pd.DataFrame, status: pd.Series, cfg: Config) -> Baseline:
    """Use the first ``cfg.baseline_days`` of healthy operation.

    Median and MAD are used instead of mean and standard deviation so that a few
    spikes in the training period don't widen the "normal" band.
    """
    healthy = healthy_mask(status, cfg)
    start = data.index[healthy.to_numpy()][0]
    end = start + pd.Timedelta(days=cfg.baseline_days)
    window = healthy & (data.index >= start) & (data.index < end)
    train = data[window.to_numpy()]

    median = train.median()
    mad = (train - median).abs().median()
    scale = (1.4826 * mad).where(lambda s: s > 0, train.std())
    scale = scale.where(scale > 0, 1.0)
    return Baseline(start=start, end=end, median=median, scale=scale)


def band(baseline: Baseline, sensor: str, k: float) -> tuple[float, float]:
    """Lower and upper limit of the expected-behaviour band for one sensor."""
    m, s = baseline.median[sensor], baseline.scale[sensor]
    return m - k * s, m + k * s
