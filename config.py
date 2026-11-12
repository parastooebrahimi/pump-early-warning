"""All tunable settings in one place, so thresholds are easy to find, document and change."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Cleaning ---
    resample_rule: str = "10min"        # aggregate 1-minute telemetry into 10-minute windows
    max_missing_fraction: float = 0.30  # drop a sensor if more than 30% of its readings are missing
    min_std: float = 1e-6               # changes smaller than this count as "unchanged"
    max_unchanged_fraction: float = 0.90  # drop a sensor stuck on the same value >90% of the time
    max_fill_gap: int = 3               # forward-fill gaps up to 3 windows (30 min); longer gaps stay missing

    # --- Baseline (expected behaviour) ---
    baseline_days: int = 14             # length of the healthy period used to learn "normal"
    exclude_before_failure_h: int = 72  # never learn "normal" from the 72 h before a failure

    # --- Detection ---
    ewma_span: int = 12                 # smoothing for the per-sensor z-score (12 windows = 2 h)
    z_threshold: float = 4.0            # a sensor is "out of band" when |smoothed z| exceeds this
    persistence: int = 3                # ...for at least 3 consecutive windows (30 min)
    min_sensors: int = 3                # raise a band alert when at least this many sensors are out of band
    cusum_k: float = 0.5                # CUSUM allowance (drift smaller than this is ignored)
    cusum_h: float = 30.0               # CUSUM decision threshold

    # --- Evaluation ---
    warning_window_h: int = 72          # an alert counts as an early warning if a failure follows within 72 h
    alert_cooldown_h: int = 12          # alerts closer together than this are treated as one alert event
    restart_grace_h: int = 6            # ignore alerts for 6 h after the pump returns to NORMAL (start-up transient)
