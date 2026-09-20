"""Operational metrics and multi-seed summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def operational_metrics(interval_log: pd.DataFrame) -> dict[str, float]:
    if interval_log.empty:
        raise ValueError("Cannot calculate metrics from an empty interval log.")
    per_interval = interval_log.sort_values("dock").groupby("time_bin", as_index=False).first()
    total_collected = float(interval_log["amount_picked"].sum())
    total_available = float(interval_log["available_before_pickup"].sum())
    total_energy = float(per_interval["total_energy_interval"].sum())
    pickup_per_interval = interval_log.groupby("time_bin")["amount_picked"].sum()
    active_per_interval = interval_log.groupby("time_bin")["active"].sum()
    active_count = int(interval_log["active"].sum())
    visited_active = int((interval_log["active"] & interval_log["visited"]).sum())
    weighted_remaining = float(
        ((interval_log["queue_after"] + interval_log["overflow"]) * interval_log["risk_probability"]).sum()
    )
    return {
        "overflow_rows": int(interval_log["overflow"].gt(0).sum()),
        "overflow_rate": float(interval_log["overflow"].gt(0).mean()),
        "total_overflow_units": float(interval_log["overflow"].sum()),
        "total_collected_units": total_collected,
        "collection_ratio": total_collected / total_available if total_available else 0.0,
        "total_route_distance": float(per_interval["total_distance_interval"].sum()),
        "total_energy_used": total_energy,
        "energy_per_collected_unit": total_energy / total_collected if total_collected else np.nan,
        "risk_weighted_remaining_goods": weighted_remaining,
        "number_of_sorties": int(per_interval["sorties_interval"].sum()),
        "active_warehouse_decisions": active_count,
        "visited_active_warehouses": visited_active,
        "unvisited_active_warehouses": active_count - visited_active,
        "active_visit_rate": visited_active / active_count if active_count else 1.0,
        "max_active_warehouses_interval": int(active_per_interval.max()),
        "intervals_with_multiple_active_warehouses": int(active_per_interval.gt(1).sum()),
        "max_route_distance_interval": float(per_interval["total_distance_interval"].max()),
        "max_energy_interval": float(per_interval["total_energy_interval"].max()),
        "max_payload_interval": float(pickup_per_interval.max()),
        "constraint_violations": int(interval_log["constraint_violations"].sum()),
        "algorithm_runtime_seconds": float(per_interval["algorithm_runtime_seconds"].sum()),
        "mean_algorithm_runtime_seconds": float(
            per_interval["algorithm_runtime_seconds"].mean()
        ),
    }


def summarize_runs(run_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column for column in run_metrics.select_dtypes(include="number").columns if column != "seed"
    ]
    summary = run_metrics.groupby("policy")[numeric].agg(["mean", "std", "min", "max", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    for metric in numeric:
        standard_error = summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"])
        summary[f"{metric}_ci95_low"] = summary[f"{metric}_mean"] - 1.96 * standard_error
        summary[f"{metric}_ci95_high"] = summary[f"{metric}_mean"] + 1.96 * standard_error
    return summary.sort_index().reset_index()
