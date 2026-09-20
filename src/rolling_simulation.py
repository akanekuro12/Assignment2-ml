"""End-to-end chronological simulation: arrivals -> MLP -> routing -> queues."""

from __future__ import annotations

import json
import time
from typing import Protocol

import numpy as np
import pandas as pd

from .aco_optimizer import optimize_aco
from .baselines import fullest_first, nearest_first
from .config_loader import warehouse_capacities, warehouse_coordinates
from .domain import drones_from_config, euclidean_distance_matrix
from .feature_builder import build_deployment_features
from .routing_utils import (
    idle_solution,
    select_active_warehouses,
)
from .solution_validator import unreachable_round_trips, validate_solution


class Predictor(Protocol):
    def predict_by_warehouse(self, features: pd.DataFrame) -> dict[int, float]: ...


SUPPORTED_POLICIES = {
    "nearest_first", "fullest_first", "aco_current", "mlp_aco",
}


def run_simulation(
    intervals: pd.DataFrame,
    config: dict,
    policy: str,
    predictor: Predictor | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; expected one of {sorted(SUPPORTED_POLICIES)}")
    if policy == "mlp_aco" and predictor is None:
        raise ValueError(f"{policy} requires a trained risk predictor.")

    warehouses = [int(node) for node in config["data"]["warehouse_ids"]]
    capacities = warehouse_capacities(config)
    distance_matrix, node_to_index, _ = euclidean_distance_matrix(warehouse_coordinates(config))
    fleet = drones_from_config(config)
    if len(fleet) != 1:
        raise ValueError("Rolling simulation requires exactly one configured drone.")
    central_node = int(config["simulation"]["central_node"])
    warnings = unreachable_round_trips(
        fleet, warehouses, distance_matrix, node_to_index, central_node
    )
    for message in warnings:
        print(f"WARNING: {message}")

    queue = {node: 0.0 for node in warehouses}
    interval_records: list[dict] = []
    route_records: list[dict] = []
    mlp = config["mlp"]
    aco = config["aco"]
    objective = config["objective"]
    base_seed = int(config["seed"] if seed is None else seed)

    for time_bin, raw_rows in intervals.groupby("time_bin", sort=True):
        rows = raw_rows.set_index("dock")
        if set(rows.index.astype(int)) != set(warehouses):
            raise AssertionError(f"Time bin {time_bin} does not contain every warehouse.")
        day = int(rows["day"].iloc[0])
        if str(config["simulation"]["battery_reset_policy"]) != "per_interval":
            raise ValueError("Single-drone simulation requires per_interval battery reset.")
        fleet[0].reset_battery()

        arrivals = {node: float(rows.loc[node, "arrivals"]) for node in warehouses}
        available = {node: queue[node] + arrivals[node] for node in warehouses}
        if predictor is not None:
            feature_rows = build_deployment_features(
                raw_rows, available, capacities, warehouses
            )
            risk = predictor.predict_by_warehouse(feature_rows)
        else:
            risk = {node: 0.0 for node in warehouses}

        use_risk = policy == "mlp_aco"
        active_nodes = select_active_warehouses(
            available=available,
            capacities=capacities,
            risk_probability=risk,
            risk_threshold=float(mlp["risk_threshold"]),
            fill_threshold=float(mlp["congestion_threshold"]),
            use_risk=use_risk,
        )
        active_goods = {node: available[node] for node in active_nodes}

        start = time.perf_counter()
        if not active_goods:
            solution = idle_solution(fleet, active_goods, central_node)
        elif policy == "nearest_first":
            solution = nearest_first(
                active_goods, fleet, distance_matrix, node_to_index,
                float(objective["unvisited_penalty"]), float(objective["distance_weight"]), central_node,
            )
        elif policy == "fullest_first":
            solution = fullest_first(
                active_goods, fleet, distance_matrix, node_to_index, capacities,
                float(objective["unvisited_penalty"]), float(objective["distance_weight"]), central_node,
            )
        else:
            interval_seed = (base_seed * 1_000_003 + int(time_bin)) % (2**32)
            solution = optimize_aco(
                active_goods, fleet, distance_matrix, node_to_index, aco,
                seed=interval_seed, central_node=central_node, objective_config=objective,
            )
        runtime = time.perf_counter() - start
        violations = validate_solution(
            solution,
            fleet,
            active_goods,
            distance_matrix,
            node_to_index,
            central_node,
            raise_on_error=False,
        )
        if violations:
            raise AssertionError("Routing policy returned an infeasible solution: " + "; ".join(violations))

        pickup_total = {node: 0.0 for node in warehouses}
        serving_drones: dict[int, list[int]] = {node: [] for node in warehouses}
        visited_nodes = {
            int(node)
            for route in solution.routes.values()
            for node in route
            if int(node) != central_node
        }
        for drone in fleet:
            for node, amount in solution.pickups.get(drone.id, {}).items():
                pickup_total[node] += float(amount)
                if amount > 0:
                    serving_drones[node].append(drone.id)
            drone.available_battery = max(
                0.0, float(drone.available_battery) - solution.energy_used.get(drone.id, 0.0)
            )
            route_records.append(
                {
                    "policy": policy,
                    "seed": base_seed,
                    "time_bin": int(time_bin),
                    "day": day,
                    "drone_id": drone.id,
                    "route": json.dumps(solution.routes[drone.id]),
                    "pickup": json.dumps(solution.pickups.get(drone.id, {}), sort_keys=True),
                    "route_distance": solution.route_distance.get(drone.id, 0.0),
                    "energy_used": solution.energy_used.get(drone.id, 0.0),
                    "battery_remaining": drone.available_battery,
                    "active_warehouse_count": len(active_nodes),
                    "visited_warehouse_count": len(visited_nodes),
                    "unvisited_warehouse_count": len(set(active_nodes) - visited_nodes),
                    "objective_value": solution.objective_value,
                }
            )

        total_energy = solution.total_energy
        total_distance = float(sum(solution.route_distance.values()))
        sortie_count = sum(distance > 0 for distance in solution.route_distance.values())
        for node in warehouses:
            raw_next = max(0.0, available[node] - pickup_total[node])
            overflow = max(0.0, raw_next - capacities[node])
            queue_after = min(float(capacities[node]), raw_next)
            interval_records.append(
                {
                    "policy": policy,
                    "seed": base_seed,
                    "time_bin": int(time_bin),
                    "day": day,
                    "hour": int(rows.loc[node, "hour"]),
                    "dock": node,
                    "arrivals": arrivals[node],
                    "queue_before": queue[node],
                    "available_before_pickup": available[node],
                    "fill_ratio": available[node] / capacities[node],
                    "risk_probability": risk[node],
                    "active": node in active_nodes,
                    "visited": node in visited_nodes,
                    "amount_picked": pickup_total[node],
                    "queue_after": queue_after,
                    "overflow": overflow,
                    "serving_drone_ids": json.dumps(serving_drones[node]),
                    "total_energy_interval": total_energy,
                    "total_distance_interval": total_distance,
                    "sorties_interval": sortie_count,
                    "objective_value": solution.objective_value,
                    "constraint_violations": len(violations),
                    "algorithm_runtime_seconds": runtime,
                }
            )
            queue[node] = queue_after

    return pd.DataFrame(interval_records), pd.DataFrame(route_records)
