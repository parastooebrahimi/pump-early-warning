"""End-to-end run: load -> clean -> baseline -> detect -> evaluate -> report."""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import plots
from .baseline import fit_baseline, failure_starts
from .clean import clean
from .config import Config
from .detect import band_alerts, cusum_alerts, health_index, smoothed_z, zscores
from .evaluate import alert_events, evaluate, monitorable
from .load import load_pump_data


def run(data_path: str | Path, out_dir: str | Path = "reports", cfg: Config = Config()) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_pump_data(data_path)
    data, status, quality = clean(raw, cfg)
    baseline = fit_baseline(data, status, cfg)

    sz = smoothed_z(zscores(data, baseline), cfg)
    band_alert, _ = band_alerts(sz, cfg)
    hi = health_index(sz)
    cusum_alert, cusum_stat = cusum_alerts(hi, cfg, reset_mask=~monitorable(status, cfg))

    results = [evaluate("Baseline band + persistence", band_alert, status, cfg),
               evaluate("CUSUM drift on health index", cusum_alert, status, cfg)]
    summary = pd.DataFrame([r.summary_row() for r in results])

    # Charts
    plots.plot_health(hi, cusum_stat, alert_events(band_alert, status, cfg), status, cfg.cusum_h,
                      out / "health_index.png")
    failures = failure_starts(status)
    if len(failures):
        pre = slice(failures[0] - pd.Timedelta(hours=cfg.warning_window_h), failures[0])
        top_sensor = sz.loc[pre].abs().mean().idxmax()
        plots.plot_sensor(data, top_sensor, baseline, cfg.z_threshold, status, out / "top_sensor_band.png")
        plots.plot_lead_window(data, sz, status, failures[0], cfg.warning_window_h, out / "before_first_failure.png")

    _write_report(out / "results.md", raw, data, quality, baseline, summary, results, cfg)
    summary.to_csv(out / "summary.csv", index=False)
    return summary


def _write_report(path, raw, data, quality, baseline, summary, results, cfg) -> None:
    lines = [
        "# Results",
        "",
        f"Data: {len(raw):,} rows from {raw.index.min():%Y-%m-%d} to {raw.index.max():%Y-%m-%d}, "
        f"resampled to {len(data):,} windows of {cfg.resample_rule}.",
        "",
        "## Data quality",
        quality.summary(),
        "",
        f"Baseline learned from {baseline.start:%Y-%m-%d} to {baseline.end:%Y-%m-%d} (healthy operation only).",
        "",
        "## Alert quality",
        summary.to_markdown(index=False),
        "",
    ]
    for r in results:
        lines += [f"### {r.detector}", r.per_failure.to_markdown(index=False), ""]
    lines += ["## Settings used", "```", *[f"{k} = {v}" for k, v in asdict(cfg).items()], "```", ""]
    path.write_text("\n".join(lines))
