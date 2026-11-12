"""Early-warning detection: deviation from baseline bands and slow drift (CUSUM)."""

import numpy as np
import pandas as pd

from .baseline import Baseline
from .config import Config


def zscores(data: pd.DataFrame, baseline: Baseline) -> pd.DataFrame:
    """How many robust standard deviations each reading is from its sensor's normal."""
    return (data - baseline.median) / baseline.scale


def smoothed_z(z: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """EWMA smoothing so a single noisy window can't trigger an alert on its own."""
    return z.ewm(span=cfg.ewma_span, min_periods=1).mean()


def rolling_features(data: pd.DataFrame, window: str = "6h") -> dict[str, pd.DataFrame]:
    """Rolling statistics used for plots and investigation (not for the alert logic)."""
    return {
        "rolling_mean": data.rolling(window).mean(),
        "rolling_std": data.rolling(window).std(),
        "rate_of_change_per_h": data.diff() * (pd.Timedelta("1h") / data.index.to_series().diff().median()),
    }


def band_alerts(sz: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.DataFrame]:
    """Alert when at least ``min_sensors`` sensors stay out of band for ``persistence`` windows.

    Returns the alert flag per window and the per-sensor out-of-band flags.
    """
    out = sz.abs() > cfg.z_threshold
    # Persistence: a sensor counts only after being out of band for N consecutive windows.
    persistent = out.rolling(cfg.persistence).sum().ge(cfg.persistence)
    alert = persistent.sum(axis=1).ge(cfg.min_sensors)
    return alert, persistent


def health_index(sz: pd.DataFrame, top_k: int = 5) -> pd.Series:
    """One number per window: the average |z| of the ``top_k`` most abnormal sensors.

    Using the top few sensors (instead of all) keeps a fault that affects a handful
    of sensors from being diluted by the many sensors that still look normal.
    """
    arr = np.sort(np.nan_to_num(sz.abs().to_numpy(), nan=0.0), axis=1)[:, -top_k:]
    return pd.Series(arr.mean(axis=1), index=sz.index, name="health_index")


def cusum_alerts(hi: pd.Series, cfg: Config, reset_mask: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """One-sided CUSUM on the health index to catch slow, sustained drift.

    ``S_t = max(0, S_{t-1} + (x_t - mu - k))``. An alert is raised when ``S_t > h``.
    The statistic resets to zero wherever ``reset_mask`` is True (e.g. while the
    pump is broken or recovering) so one failure doesn't leak into the next.
    """
    mu = hi.median()
    s = np.zeros(len(hi))
    values = hi.to_numpy()
    resets = reset_mask.to_numpy() if reset_mask is not None else np.zeros(len(hi), dtype=bool)
    for i in range(1, len(values)):
        s[i] = 0.0 if resets[i] else max(0.0, s[i - 1] + values[i] - mu - cfg.cusum_k)
    stat = pd.Series(s, index=hi.index, name="cusum")
    return stat > cfg.cusum_h, stat
