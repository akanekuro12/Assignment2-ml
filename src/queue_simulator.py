"""Historical drone-only queue generation for leakage-safe MLP training."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .baselines import fullest_first
from .domain import DroneState
from .routing_utils import select_active_warehouses


def simulate_historical_queue(
    intervals: pd.DataFrame,
    capacities: Mapping[int, int],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    fill_threshold: float,
    unvisited_penalty: float,
    distance_weight: float,
    central_node: int = 0,
    battery_reset_policy: str = "per_interval",
) -> pd.DataFrame:
    """Generate states with a fixed fullest-first policy and no future information."""
    warehouses = sorted(int(node) for node in capacities)
    queue = {node: 0.0 for node in warehouses}
    fleet = [drone.copy() for drone in drones]
    records: list[dict] = []
    if len(fleet) != 1:
        raise ValueError("Historical queue simulation requires exactly one drone.")

    for time_bin, rows in intervals.groupby("time_bin", sort=True):
        rows = rows.set_index("dock")
        day = int(rows["day"].iloc[0])
        if battery_reset_policy != "per_interval":
            raise ValueError("Single-drone simulation requires per_interval battery reset.")
        fleet[0].reset_battery()

        arrivals = {node: float(rows.loc[node, "arrivals"]) for node in warehouses}
        available = {node: queue[node] + arrivals[node] for node in warehouses}
        zero_risk = {node: 0.0 for node in warehouses}
        active_nodes = select_active_warehouses(
            available=available,
            capacities=capacities,
            risk_probability=zero_risk,
            risk_threshold=1.1,
            fill_threshold=fill_threshold,
            use_risk=False,
        )
        active_goods = {node: available[node] for node in active_nodes}
        solution = fullest_first(
            active_goods=active_goods,
            drones=fleet,
            distance_matrix=distance_matrix,
            node_to_index=node_to_index,
            capacities=capacities,
            unvisited_penalty=unvisited_penalty,
            distance_weight=distance_weight,
            central_node=central_node,
        )

        pickup_total = {node: 0.0 for node in warehouses}
        for drone in fleet:
            for node, amount in solution.pickups.get(drone.id, {}).items():
                pickup_total[node] += float(amount)
            drone.available_battery = max(
                0.0, float(drone.available_battery) - solution.energy_used.get(drone.id, 0.0)
            )

        for node in warehouses:
            raw_next = max(0.0, available[node] - pickup_total[node])
            overflow = max(0.0, raw_next - capacities[node])
            queue[node] = min(float(capacities[node]), raw_next)
            records.append(
                {
                    "time_bin": int(time_bin),
                    "interval_start_seconds": int(rows.loc[node, "interval_start_seconds"]),
                    "day": day,
                    "hour": int(rows.loc[node, "hour"]),
                    "dock": node,
                    "arrivals_15m": int(arrivals[node]),
                    "mean_arrivals_60m": rows.loc[node, "mean_arrivals_60m"],
                    "queue_before": float(available[node] - arrivals[node]),
                    "available_before_pickup": float(available[node]),
                    "amount_picked": float(pickup_total[node]),
                    "queue_after": float(queue[node]),
                    "overflow": float(overflow),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(["time_bin", "dock"]).reset_index(drop=True)
