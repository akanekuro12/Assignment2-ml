"""Exhaustive small-instance reference solver for up to four active warehouses."""

from __future__ import annotations

from itertools import permutations, product
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from .domain import DroneState, RoutingSolution, route_distance
from .routing_utils import calculate_objective, idle_solution
from .solution_validator import validate_solution


def _feasible_routes(
    drone: DroneState,
    active_nodes: list[int],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    central_node: int,
) -> list[list[int]]:
    routes = [[central_node, central_node]]
    for length in range(1, len(active_nodes) + 1):
        for order in permutations(active_nodes, length):
            route = [central_node, *order, central_node]
            energy = route_distance(route, distance_matrix, node_to_index) * drone.energy_per_km
            if energy <= float(drone.available_battery) + 1e-9:
                routes.append(route)
    return routes


def solve_exact_small_instance(
    demand: Mapping[int, float],
    urgency: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    weight_remaining: float = 0.7,
    weight_energy: float = 0.3,
    central_node: int = 0,
) -> RoutingSolution:
    """Enumerate routes and solve the optimal continuous pickup allocation by LP."""
    if not demand:
        return idle_solution(drones, demand, central_node)
    nodes = sorted(int(node) for node in demand)
    if len(nodes) > 4:
        raise ValueError("The exact reference solver is intentionally limited to four warehouses.")
    route_options = [
        _feasible_routes(drone, nodes, distance_matrix, node_to_index, central_node)
        for drone in drones
    ]
    best: RoutingSolution | None = None
    variable_count = len(drones) * len(nodes)
    demand_total = sum(float(urgency[node]) * float(demand[node]) for node in nodes) + 1e-12

    for route_combination in product(*route_options):
        energy = {
            drone.id: route_distance(route_combination[index], distance_matrix, node_to_index)
            * drone.energy_per_km
            for index, drone in enumerate(drones)
        }
        c = np.zeros(variable_count, dtype=float)
        bounds = []
        for drone_index, route in enumerate(route_combination):
            visited = set(route)
            for node_index, node in enumerate(nodes):
                variable = drone_index * len(nodes) + node_index
                c[variable] = -weight_remaining * float(urgency[node]) / demand_total
                bounds.append((0.0, float(demand[node]) if node in visited else 0.0))

        a_ub, b_ub = [], []
        for drone_index, drone in enumerate(drones):
            row = np.zeros(variable_count)
            row[drone_index * len(nodes):(drone_index + 1) * len(nodes)] = 1.0
            a_ub.append(row)
            b_ub.append(float(drone.payload_capacity))
        for node_index, node in enumerate(nodes):
            row = np.zeros(variable_count)
            row[node_index::len(nodes)] = 1.0
            a_ub.append(row)
            b_ub.append(float(demand[node]))
        allocation = linprog(
            c,
            A_ub=np.asarray(a_ub),
            b_ub=np.asarray(b_ub),
            bounds=bounds,
            method="highs",
        )
        if not allocation.success:
            continue
        pickups = {drone.id: {} for drone in drones}
        for drone_index, drone in enumerate(drones):
            for node_index, node in enumerate(nodes):
                amount = float(allocation.x[drone_index * len(nodes) + node_index])
                if amount > 1e-9:
                    pickups[drone.id][node] = amount
        remaining = {
            node: float(demand[node]) - sum(values.get(node, 0.0) for values in pickups.values())
            for node in nodes
        }
        distances = {
            drone.id: route_distance(route_combination[index], distance_matrix, node_to_index)
            for index, drone in enumerate(drones)
        }
        objective = calculate_objective(
            remaining, demand, urgency, energy, drones, weight_remaining, weight_energy
        )
        candidate = RoutingSolution(
            routes={drone.id: list(route_combination[index]) for index, drone in enumerate(drones)},
            pickups=pickups,
            route_distance=distances,
            energy_used=energy,
            remaining_goods=remaining,
            objective_value=objective,
        )
        if best is None or candidate.objective_value < best.objective_value:
            best = candidate
    if best is None:
        raise RuntimeError("No feasible exact solution was found.")
    validate_solution(best, drones, demand, distance_matrix, node_to_index, central_node)
    return best
