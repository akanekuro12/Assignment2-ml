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
        "constraint_violations": int(interval_log["constraint_violations"].sum()),
        "aco_runtime_seconds": float(per_interval["algorithm_runtime_seconds"].sum()),
    }


def summarize_runs(run_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column for column in run_metrics.select_dtypes(include="number").columns if column != "seed"
    ]
    return (
        run_metrics.groupby("policy")[numeric]
        .agg(["mean", "std", "min"])
        .sort_index()
    )
