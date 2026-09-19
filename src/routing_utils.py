"""Shared calculations for constrained single-drone routing."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .domain import DroneState, RoutingSolution, route_distance


EPSILON = 1e-12


def select_active_warehouses(
    available: Mapping[int, float],
    capacities: Mapping[int, float],
    risk_probability: Mapping[int, float],
    risk_threshold: float,
    fill_threshold: float,
    use_risk: bool,
) -> list[int]:
    """Filter warehouses before routing; low-risk nodes never enter MLP-ACO."""
    active: list[int] = []
    for node, amount in available.items():
        if float(amount) <= EPSILON:
            continue
        if use_risk:
            selected = float(risk_probability.get(node, 0.0)) >= float(risk_threshold)
        else:
            selected = float(amount) / float(capacities[node]) >= float(fill_threshold)
        if selected:
            active.append(int(node))
    return sorted(active)


def route_limits(drone: DroneState) -> tuple[float, float]:
    """Return explicit distance limit and battery-implied distance limit."""
    explicit = float(drone.max_route_distance_km)
    battery_distance = float(drone.available_battery) / float(drone.energy_per_km)
    return explicit, battery_distance


def route_is_feasible(
    route: Sequence[int],
    drone: DroneState,
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    tolerance: float = 1e-9,
) -> bool:
    distance = route_distance(list(route), distance_matrix, node_to_index)
    energy = distance * float(drone.energy_per_km)
    return bool(
        distance <= float(drone.max_route_distance_km) + tolerance
        and energy <= float(drone.available_battery) + tolerance
    )


def allocate_pickups(
    route: Sequence[int],
    available: Mapping[int, float],
    drone: DroneState,
    central_node: int = 0,
) -> tuple[dict[int, float], dict[int, float]]:
    """Share the payload proportionally across the active warehouses on the route."""
    remaining = {int(node): float(amount) for node, amount in available.items()}
    visited = [
        int(node)
        for node in route
        if int(node) != central_node and int(node) in remaining
    ]
    total_available = sum(remaining[node] for node in visited)
    payload = min(float(drone.payload_capacity), total_available)
    pickups: dict[int, float] = {}
    if payload <= EPSILON or total_available <= EPSILON:
        return pickups, remaining

    for node in visited:
        amount = payload * remaining[node] / total_available
        if amount > EPSILON:
            pickups[node] = amount
            remaining[node] -= amount
    return pickups, remaining


def calculate_route_objective(
    route: Sequence[int],
    active_warehouses: Sequence[int],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    unvisited_penalty: float,
    distance_weight: float,
    central_node: int = 0,
) -> float:
    visited = {int(node) for node in route if int(node) != central_node}
    unvisited_count = len(set(int(node) for node in active_warehouses) - visited)
    distance = route_distance(list(route), distance_matrix, node_to_index)
    return float(unvisited_penalty * unvisited_count + distance_weight * distance)


def idle_solution(
    drones: Sequence[DroneState], demand: Mapping[int, float], central_node: int = 0
) -> RoutingSolution:
    return RoutingSolution(
        routes={drone.id: [central_node, central_node] for drone in drones},
        pickups={drone.id: {} for drone in drones},
        route_distance={drone.id: 0.0 for drone in drones},
        energy_used={drone.id: 0.0 for drone in drones},
        remaining_goods={int(node): float(value) for node, value in demand.items()},
        objective_value=0.0 if not demand else float("inf"),
    )


def build_route_solution(
    route: list[int],
    active_goods: Mapping[int, float],
    active_warehouses: Sequence[int],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    unvisited_penalty: float,
    distance_weight: float,
    central_node: int = 0,
) -> RoutingSolution:
    if len(drones) != 1:
        raise ValueError("Single-drone routing requires exactly one drone.")
    drone = drones[0]
    pickups, remaining = allocate_pickups(route, active_goods, drone, central_node)
    distance = route_distance(route, distance_matrix, node_to_index)
    energy = distance * float(drone.energy_per_km)
    objective = calculate_route_objective(
        route,
        active_warehouses,
        distance_matrix,
        node_to_index,
        unvisited_penalty,
        distance_weight,
        central_node,
    )
    return RoutingSolution(
        routes={drone.id: list(route)},
        pickups={drone.id: pickups},
        route_distance={drone.id: distance},
        energy_used={drone.id: energy},
        remaining_goods=remaining,
        objective_value=objective,
    )
