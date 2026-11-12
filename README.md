# Pump Failure Early Warning

Time-series analytics in Python that looks for early signs of failure in industrial pump sensor data, then measures how much warning it would have given before each recorded breakdown.

The aim is analytics a maintenance team could trust and act on. It uses simple, explainable methods (baseline bands and drift detection) rather than a black-box model, and every threshold is documented and adjustable.

## The data

[Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) (Kaggle): a water pump with 52 sensors recorded every minute from April to August 2018, with each minute labelled `NORMAL`, `BROKEN` or `RECOVERING`. It has 7 recorded failures.

Like most real telemetry, it is messy. One sensor is completely empty, some have long gaps, and readings shift when the pump restarts after repair.

## How it works

1. **Data quality** (`clean.py`)
   - Drops sensors with more than 30% missing data, and sensors stuck on one value.
   - Resamples 1-minute readings into 10-minute windows.
   - Fills gaps of up to 30 minutes. Longer gaps stay missing instead of being invented.

2. **Expected behaviour** (`baseline.py`)
   - Learns each sensor's normal range from 14 days of healthy operation.
   - Leaves out the 72 hours before any failure, so early fault behaviour isn't learned as "normal".
   - Uses median and MAD rather than mean and standard deviation, so a few spikes don't widen the band.

3. **Detection** (`detect.py`). Two detectors are compared:
   - **Baseline band + persistence.** Each reading is converted to a robust z-score and smoothed with an EWMA. An alert is raised when at least 3 sensors stay more than 4 robust SDs from normal for 30 minutes in a row.
   - **CUSUM drift.** Applies a cumulative sum to a single health index (the average deviation of the 5 most abnormal sensors) to catch slow, sustained drift that no single reading would flag.

4. **Alert quality** (`evaluate.py`)
   - For each recorded failure: was there a warning in the 72 hours before it, and how many hours of lead time did it give?
   - Any alert not followed by a failure counts as a false alarm.
   - Alerts in the 6 hours after a restart are ignored, because readings are still settling.

All settings live in `src/early_warning/config.py`.

## Running it

```bash
pip install -r requirements.txt

# Download sensor.csv from Kaggle into data/, then:
python run_analysis.py --data data/sensor.csv --out reports

# No data yet? Generate a synthetic file with the same layout:
python scripts/make_sample_data.py data/sample_sensor.csv
python run_analysis.py --data data/sample_sensor.csv --out reports/sample

# Tests
python -m pytest
```

The run writes `results.md`, `summary.csv` and three charts to the output folder:
- the health index and CUSUM over time, with failures marked;
- the most informative sensor against its expected-behaviour band;
- a zoom into the hours before the first failure.

## Results

_Run on the Kaggle data and paste the summary table from `reports/results.md` here._

## Limitations and next steps

- **Thresholds are fixed.** They are set by judgement, not tuned on held-out failures. With only 7 failures there is a real risk of overfitting if tuned too hard.
- **One baseline for the whole period.** A production version would recalibrate as operating conditions change.
- **No physical meaning yet.** The sensors are anonymised. With real tag names, alerts could be linked to failure modes (for example bearing wear or cavitation) and to a recommended action.
- **Next steps.** Load cleaned data and alerts into a SQL database and add a dashboard for maintenance planners.
