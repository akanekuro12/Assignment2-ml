"""Leakage-safe MLP features and chronological train/validation/test splits."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


NUMERIC_FEATURES = ["buffer_fill_ratio", "arrivals_15m", "mean_arrivals_60m"]


def dock_feature_names(warehouse_ids: list[int]) -> list[str]:
    return [f"dock_{int(node)}" for node in warehouse_ids]


def feature_names(warehouse_ids: list[int]) -> list[str]:
    return NUMERIC_FEATURES + dock_feature_names(warehouse_ids)


def add_no_drone_labels(
    state_table: pd.DataFrame,
    capacities: Mapping[int, float],
    horizon_intervals: int,
    congestion_threshold: float,
) -> pd.DataFrame:
    """Label whether no-drone stock reaches the threshold in the next horizon."""
    frame = state_table.sort_values(["dock", "time_bin"]).copy()
    future_arrivals = []
    grouped = frame.groupby("dock", sort=False)["arrivals_15m"]
    for step in range(1, horizon_intervals + 1):
        shifted = [grouped.shift(-offset) for offset in range(1, step + 1)]
        cumulative = sum(shifted)
        future_arrivals.append(frame["available_before_pickup"] + cumulative)
    future_values = pd.concat(future_arrivals, axis=1)
    # skipna=False makes the last H rows in each warehouse unusable, as required.
    frame["future_no_drone_max"] = future_values.max(axis=1, skipna=False)
    capacity = frame["dock"].map({int(key): float(value) for key, value in capacities.items()})
    frame["congestion"] = (
        frame["future_no_drone_max"] >= capacity * float(congestion_threshold)
    ).astype(np.int8)
    return frame.sort_values(["time_bin", "dock"]).reset_index(drop=True)


def build_model_table(
    state_table: pd.DataFrame,
    capacities: Mapping[int, float],
    warehouse_ids: list[int],
    horizon_intervals: int,
    congestion_threshold: float,
) -> pd.DataFrame:
    frame = add_no_drone_labels(
        state_table,
        capacities,
        horizon_intervals,
        congestion_threshold,
    )
    frame["buffer_fill_ratio"] = frame["available_before_pickup"] / frame["dock"].map(
        {int(key): float(value) for key, value in capacities.items()}
    )
    dummies = pd.get_dummies(frame["dock"], prefix="dock", dtype=np.int8)
    dummies = dummies.reindex(columns=dock_feature_names(warehouse_ids), fill_value=0)
    frame = pd.concat([frame, dummies], axis=1)
    return frame.dropna(subset=["mean_arrivals_60m", "future_no_drone_max"]).reset_index(drop=True)


def build_deployment_features(
    rows: pd.DataFrame,
    available: Mapping[int, float],
    capacities: Mapping[int, float],
    warehouse_ids: list[int],
) -> pd.DataFrame:
    indexed = rows.set_index("dock")
    records = []
    for node in warehouse_ids:
        record = {
            "buffer_fill_ratio": float(available[node]) / float(capacities[node]),
            "arrivals_15m": float(indexed.loc[node, "arrivals"]),
            "mean_arrivals_60m": float(indexed.loc[node, "mean_arrivals_60m"]),
        }
        record.update({f"dock_{other}": int(node == other) for other in warehouse_ids})
        records.append(record)
    return pd.DataFrame(records, index=warehouse_ids)[feature_names(warehouse_ids)]


def chronological_masks(
    model_table: pd.DataFrame,
    train_end_day: int,
    validation_end_day: int,
    horizon_intervals: int,
    interval_seconds: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_boundary = int(train_end_day) * 86_400
    validation_boundary = int(validation_end_day) * 86_400
    horizon_seconds = int(horizon_intervals) * int(interval_seconds)
    times = model_table["interval_start_seconds"]
    train = times.add(horizon_seconds).lt(train_boundary)
    validation = times.ge(train_boundary) & times.add(horizon_seconds).lt(validation_boundary)
    test = times.ge(validation_boundary)
    if (train & validation).any() or (train & test).any() or (validation & test).any():
        raise AssertionError("Chronological data splits overlap.")
    return train, validation, test
