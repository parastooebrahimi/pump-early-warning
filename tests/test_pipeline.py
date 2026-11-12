import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from early_warning import Config, run  # noqa: E402
from early_warning.clean import clean  # noqa: E402
from early_warning.detect import cusum_alerts  # noqa: E402
from early_warning.evaluate import evaluate  # noqa: E402
from early_warning.load import load_pump_data  # noqa: E402
from make_sample_data import make  # noqa: E402


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    make().to_csv(path)
    return path


def test_cleaning_drops_bad_sensors(sample_csv):
    data, status, report = clean(load_pump_data(sample_csv), Config())
    assert "sensor_15" in report.dropped_missing   # all missing
    assert "sensor_50" in report.dropped_missing   # 40% missing
    assert "sensor_33" in report.dropped_flat      # stuck value
    assert data.index.freq is not None or data.index.to_series().diff().nunique() == 1
    assert set(status.unique()) <= {"NORMAL", "BROKEN", "RECOVERING"}


def test_short_gaps_filled_long_gaps_kept():
    idx = pd.date_range("2020-01-01", periods=10 * 6 * 10, freq="1min")
    df = pd.DataFrame({"sensor_00": np.arange(len(idx), dtype=float),
                       "sensor_01": np.random.default_rng(0).normal(size=len(idx))}, index=idx)
    df.iloc[100:120, 0] = np.nan   # 20 min gap -> filled
    df.iloc[300:400, 0] = np.nan   # 100 min gap -> partly left missing
    df["machine_status"] = "NORMAL"
    data, _, _ = clean(df, Config(max_missing_fraction=0.5))
    assert data["sensor_00"].isna().sum() > 0
    assert data.loc["2020-01-01 01:40":"2020-01-01 02:00", "sensor_00"].notna().all()


def test_cusum_detects_step_and_resets():
    idx = pd.date_range("2020-01-01", periods=200, freq="10min")
    hi = pd.Series(1.0, index=idx)
    hi.iloc[100:] = 3.0
    alert, stat = cusum_alerts(hi, Config(cusum_k=0.5, cusum_h=5))
    assert not alert.iloc[:100].any()
    assert alert.iloc[110:].all()
    reset = pd.Series(False, index=idx)
    reset.iloc[150] = True
    _, stat2 = cusum_alerts(hi, Config(cusum_k=0.5, cusum_h=5), reset)
    assert stat2.iloc[150] == 0


def test_evaluation_counts_lead_time_and_false_alarms():
    idx = pd.date_range("2020-01-01", periods=24 * 6 * 10, freq="10min")
    status = pd.Series("NORMAL", index=idx)
    status["2020-01-05 00:00":"2020-01-05 02:00"] = "BROKEN"
    alert = pd.Series(False, index=idx)
    alert["2020-01-04 06:00"] = True   # 18 h before the failure -> early warning
    alert["2020-01-08 12:00"] = True   # no failure follows -> false alarm
    ev = evaluate("test", alert, status, Config(warning_window_h=72))
    assert ev.per_failure["lead_time_h"].iloc[0] == 18.0
    assert ev.false_alarms == 1


def test_end_to_end_warns_before_every_synthetic_failure(sample_csv, tmp_path):
    summary = run(sample_csv, tmp_path, Config())
    band = summary.iloc[0]
    assert band["failures"] == 3
    assert band["detected"] == 3
    assert band["median_lead_time_h"] > 0
    assert band["false_alarms"] == 0
    for f in ["results.md", "summary.csv", "health_index.png", "top_sensor_band.png", "before_first_failure.png"]:
        assert (tmp_path / f).exists()


def test_alerts_ignored_during_restart_grace_period():
    idx = pd.date_range("2020-01-01", periods=24 * 6 * 4, freq="10min")
    status = pd.Series("NORMAL", index=idx)
    status["2020-01-02 00:00":"2020-01-02 05:50"] = "RECOVERING"
    alert = pd.Series(False, index=idx)
    alert["2020-01-02 07:00"] = True   # 1 h after restart -> ignored
    alert["2020-01-03 12:00"] = True   # well after restart -> counted
    ev = evaluate("test", alert, status, Config(restart_grace_h=6))
    assert ev.alert_events == 1
