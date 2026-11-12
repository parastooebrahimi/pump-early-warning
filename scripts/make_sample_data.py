"""Generate a small synthetic file with the same layout as the Kaggle pump dataset.

It exists so the pipeline and tests can run without downloading the real data.
Results on this file are NOT real findings; use the Kaggle data for those.

Usage: python scripts/make_sample_data.py data/sample_sensor.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(days: int = 45, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-04-01", periods=days * 24 * 60, freq="1min")
    n = len(idx)
    t_days = np.arange(n) / (24 * 60)

    sensors = {}
    for i in range(52):
        level = rng.uniform(10, 600)
        noise = level * rng.uniform(0.01, 0.04)
        daily = level * 0.02 * np.sin(2 * np.pi * t_days + rng.uniform(0, 6))
        sensors[f"sensor_{i:02d}"] = level + daily + rng.normal(0, noise, n)
    df = pd.DataFrame(sensors, index=idx)
    status = np.array(["NORMAL"] * n, dtype=object)

    # Three failures. Some sensors drift away from normal in the 36 h before each one.
    for day, affected in [(20, [4, 10, 11, 12]), (32, [2, 4, 38]), (41, [10, 11, 25, 38])]:
        f = day * 24 * 60
        ramp = 36 * 60
        for s in affected:
            col = f"sensor_{s:02d}"
            sd = df[col].iloc[: 10 * 24 * 60].std()
            df.iloc[f - ramp:f, df.columns.get_loc(col)] += np.linspace(0, 8 * sd, ramp)
        status[f:f + 120] = "BROKEN"
        status[f + 120:f + 120 + 12 * 60] = "RECOVERING"
        df.iloc[f:f + 120 + 12 * 60] *= 0.3

    # Real telemetry problems: a dead sensor, a flat sensor, a patchy sensor, gaps and spikes.
    df["sensor_15"] = np.nan
    df["sensor_33"] = 42.0
    df.loc[rng.random(n) < 0.4, "sensor_50"] = np.nan
    for _ in range(40):
        start = rng.integers(0, n - 60)
        df.iloc[start:start + rng.integers(1, 60), rng.integers(0, 52)] = np.nan
    spikes = rng.random(df.shape) < 0.0005
    df = df.mask(spikes, df * 3)

    out = df.reset_index().rename(columns={"index": "timestamp"})
    out["machine_status"] = status
    return out


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample_sensor.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(path)
    print(f"Wrote {path}")
