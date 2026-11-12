"""Measure alert quality against the recorded failures: lead time, missed failures, false alarms."""

from dataclasses import dataclass

import pandas as pd

from .baseline import failure_starts
from .config import Config


@dataclass
class Evaluation:
    detector: str
    per_failure: pd.DataFrame  # one row per failure: when it happened, first warning, lead time
    false_alarms: int
    alert_events: int

    @property
    def detected(self) -> int:
        return int(self.per_failure["lead_time_h"].notna().sum())

    def summary_row(self) -> dict:
        lead = self.per_failure["lead_time_h"].dropna()
        return {
            "detector": self.detector,
            "failures": len(self.per_failure),
            "detected": self.detected,
            "missed": len(self.per_failure) - self.detected,
            "median_lead_time_h": round(float(lead.median()), 1) if len(lead) else None,
            "alert_events": self.alert_events,
            "false_alarms": self.false_alarms,
        }


def monitorable(status: pd.Series, cfg: Config) -> pd.Series:
    """True where the pump is NORMAL and past the start-up grace period after a restart."""
    normal = status.eq("NORMAL")
    restarts = status.index[normal & ~normal.shift(fill_value=True)]
    mask = normal.copy()
    for t in restarts:
        mask &= ~((status.index >= t) & (status.index < t + pd.Timedelta(hours=cfg.restart_grace_h)))
    return mask


def alert_events(alert: pd.Series, status: pd.Series, cfg: Config) -> pd.DatetimeIndex:
    """Turn a per-window alert flag into distinct alert events.

    Only alerts raised while the pump is NORMAL count (an alert while it's already
    broken is not a warning), and not in the first hours after a restart, when
    readings are still settling. Alerts within the cooldown are merged into one event.
    """
    times = alert[alert & monitorable(status, cfg)].index
    events, last = [], None
    for t in times:
        if last is None or t - last > pd.Timedelta(hours=cfg.alert_cooldown_h):
            events.append(t)
        last = t
    return pd.DatetimeIndex(events)


def evaluate(name: str, alert: pd.Series, status: pd.Series, cfg: Config) -> Evaluation:
    window = pd.Timedelta(hours=cfg.warning_window_h)
    failures = failure_starts(status)
    events = alert_events(alert, status, cfg)

    rows, used = [], set()
    for f in failures:
        before = [e for e in events if f - window <= e < f]
        first = before[0] if before else None
        used.update(before)
        rows.append({
            "failure_at": f,
            "first_warning_at": first,
            "lead_time_h": round((f - first).total_seconds() / 3600, 1) if first is not None else None,
        })

    per_failure = pd.DataFrame(rows, columns=["failure_at", "first_warning_at", "lead_time_h"])
    false_alarms = sum(1 for e in events if e not in used)
    return Evaluation(name, per_failure, false_alarms, len(events))
