"""Load the pump sensor CSV into a time-indexed DataFrame."""

from pathlib import Path

import pandas as pd

STATUS_ORDER = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 2}


def load_pump_data(path: str | Path) -> pd.DataFrame:
    """Read the Kaggle pump sensor file.

    Expected columns: ``timestamp``, ``sensor_00`` ... ``sensor_51`` and ``machine_status``
    (NORMAL / BROKEN / RECOVERING). An unnamed index column, if present, is dropped.
    """
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    unknown = set(df["machine_status"].unique()) - set(STATUS_ORDER)
    if unknown:
        raise ValueError(f"Unexpected machine_status values: {unknown}")
    return df


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]
