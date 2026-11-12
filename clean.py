"""Data quality: remove unusable sensors, resample and handle gaps."""

from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .load import STATUS_ORDER, sensor_columns


@dataclass
class QualityReport:
    dropped_missing: list[str] = field(default_factory=list)
    dropped_flat: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    missing_after_resample: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Kept {len(self.kept)} sensors. "
            f"Dropped {len(self.dropped_missing)} for missing data {self.dropped_missing} "
            f"and {len(self.dropped_flat)} as flat/dead {self.dropped_flat}."
        )


def clean(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, QualityReport]:
    """Return (sensor data at cfg.resample_rule, machine status per window, quality report)."""
    report = QualityReport()
    sensors = sensor_columns(df)

    # 1. Sensors with too much missing data can't be monitored reliably.
    missing = df[sensors].isna().mean()
    report.dropped_missing = sorted(missing[missing > cfg.max_missing_fraction].index)
    sensors = [s for s in sensors if s not in report.dropped_missing]

    # 2. Stuck sensors repeat the same value almost all the time and carry no information.
    #    (Checking the share of unchanged readings is more robust than standard deviation,
    #    which a few spikes on a stuck sensor would inflate.)
    unchanged = df[sensors].diff().abs().lt(cfg.min_std).mean()
    report.dropped_flat = sorted(unchanged[unchanged > cfg.max_unchanged_fraction].index)
    sensors = [s for s in sensors if s not in report.dropped_flat]
    report.kept = sensors

    # 3. Resample onto a regular time grid; the mean smooths out minute-level noise.
    data = df[sensors].resample(cfg.resample_rule).mean()

    # 4. Fill short gaps only. Long gaps stay missing rather than inventing data.
    data = data.ffill(limit=cfg.max_fill_gap)
    report.missing_after_resample = data.isna().mean().round(4).to_dict()

    # 5. Machine status per window = the worst status seen in that window.
    codes = df["machine_status"].map(STATUS_ORDER)
    status = codes.resample(cfg.resample_rule).max().reindex(data.index).ffill()
    inverse = {v: k for k, v in STATUS_ORDER.items()}
    status = status.map(inverse)

    return data, status, report
