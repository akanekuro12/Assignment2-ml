"""WEPAStacks event cleaning and 15-minute arrival aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config_loader import PROJECT_ROOT, resolve_project_path


def resolve_csv_path(config: dict) -> Path:
    configured = Path(config["data"]["csv_path"])
    candidates = [
        resolve_project_path(configured, config),
        PROJECT_ROOT / configured.name,
        PROJECT_ROOT / "data" / configured.name,
        PROJECT_ROOT / "assignment_2" / configured.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = "\n - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {configured.name}. Searched:\n - {searched}")


def clean_events(raw_df: pd.DataFrame, warehouse_ids: Iterable[int]) -> pd.DataFrame:
    frame = raw_df.copy()
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    required = {"arrival_time", "dock"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if "type" in frame.columns:
        event_type = frame["type"].astype(str).str.strip().str.lower()
        frame = frame.loc[event_type.eq("delivery")].copy()

    frame["arrival_time"] = pd.to_numeric(frame["arrival_time"], errors="coerce")
    frame["dock"] = pd.to_numeric(frame["dock"], errors="coerce")
    frame = frame.dropna(subset=["arrival_time", "dock"]).copy()
    frame["arrival_time"] = frame["arrival_time"].astype(np.int64)
    frame["dock"] = frame["dock"].astype(np.int64)
    allowed = {int(value) for value in warehouse_ids}
    frame = frame.loc[frame["arrival_time"].ge(0) & frame["dock"].isin(allowed)].copy()
    return frame.sort_values(["arrival_time", "dock"]).reset_index(drop=True)


def aggregate_intervals(
    clean_df: pd.DataFrame,
    warehouse_ids: Iterable[int],
    interval_seconds: int = 900,
) -> pd.DataFrame:
    if clean_df.empty:
        raise ValueError("No valid inbound events remain after cleaning.")
    warehouses = [int(value) for value in warehouse_ids]
    frame = clean_df.copy()
    frame["time_bin"] = np.floor_divide(frame["arrival_time"], interval_seconds).astype(np.int64)
    grouped = frame.groupby(["time_bin", "dock"]).size().rename("arrivals").astype(np.int64)
    all_bins = np.arange(frame["time_bin"].min(), frame["time_bin"].max() + 1, dtype=np.int64)
    full_index = pd.MultiIndex.from_product([all_bins, warehouses], names=["time_bin", "dock"])
    intervals = grouped.reindex(full_index, fill_value=0).reset_index()
    intervals["arrivals"] = intervals["arrivals"].astype(np.int64)
    intervals["interval_start_seconds"] = intervals["time_bin"] * int(interval_seconds)
    intervals["day"] = intervals["interval_start_seconds"] // 86_400 + 1
    intervals["hour"] = (intervals["interval_start_seconds"] % 86_400) // 3_600
    intervals = intervals.sort_values(["time_bin", "dock"]).reset_index(drop=True)

    counts = intervals.groupby("time_bin")["dock"].nunique()
    if not counts.eq(len(warehouses)).all():
        raise AssertionError("Every time bin must contain every warehouse.")
    if intervals["arrivals"].isna().any() or intervals["arrivals"].lt(0).any():
        raise AssertionError("Arrivals must be non-negative integers without missing values.")
    return intervals


def add_arrival_history(intervals: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    frame = intervals.sort_values(["dock", "time_bin"]).copy()
    frame["mean_arrivals_60m"] = frame.groupby("dock", sort=False)["arrivals"].transform(
        lambda values: values.rolling(window=window, min_periods=window).mean()
    )
    return frame.sort_values(["time_bin", "dock"]).reset_index(drop=True)


def load_interval_table(config: dict) -> pd.DataFrame:
    path = resolve_csv_path(config)
    raw = pd.read_csv(path)
    warehouses = config["data"]["warehouse_ids"]
    clean = clean_events(raw, warehouses)
    intervals = aggregate_intervals(
        clean,
        warehouses,
        interval_seconds=int(config["data"]["interval_seconds"]),
    )
    return add_arrival_history(intervals, window=4)
