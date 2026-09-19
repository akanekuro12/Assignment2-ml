"""Constrained ACO for one drone and a filtered set of warehouses."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .domain import DroneState, RoutingSolution
from .routing_utils import EPSILON, build_route_solution, idle_solution
from .solution_validator import validate_solution


def _construct_ant_solution(
    active_goods: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    pheromone: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    unvisited_penalty: float,
    distance_weight: float,
    central_node: int,
) -> RoutingSolution:
    if len(drones) != 1:
        raise ValueError("Single-drone ACO requires exactly one drone.")
    drone = drones[0]
    active_nodes = sorted(int(node) for node in active_goods)
    unvisited = set(active_nodes)
    current = central_node
    route = [central_node]
    distance_to_current = 0.0

    while unvisited:
        feasible: list[int] = []
        for node in sorted(unvisited):
            travel = distance_matrix[node_to_index[current], node_to_index[node]]
            return_distance = distance_matrix[node_to_index[node], node_to_index[central_node]]
            projected_distance = distance_to_current + travel + return_distance
            projected_energy = projected_distance * float(drone.energy_per_km)
            if (
                projected_distance <= float(drone.max_route_distance_km) + EPSILON
                and projected_energy <= float(drone.available_battery) + EPSILON
            ):
                feasible.append(node)
        if not feasible:
            break

        scores = []
        for node in feasible:
            distance = distance_matrix[node_to_index[current], node_to_index[node]]
            heuristic = 1.0 / (EPSILON + distance)
            tau = pheromone[node_to_index[current], node_to_index[node]]
            scores.append(max(EPSILON, tau) ** alpha * heuristic ** beta)
        probabilities = np.asarray(scores, dtype=float)
        probabilities /= probabilities.sum()
        node = int(rng.choice(np.asarray(feasible, dtype=int), p=probabilities))

        distance_to_current += distance_matrix[node_to_index[current], node_to_index[node]]
        route.append(node)
        unvisited.remove(node)
        current = node

    route.append(central_node)
    return build_route_solution(
        route=route,
        active_goods=active_goods,
        active_warehouses=active_nodes,
        drones=drones,
        distance_matrix=distance_matrix,
        node_to_index=node_to_index,
        unvisited_penalty=unvisited_penalty,
        distance_weight=distance_weight,
        central_node=central_node,
    )


def optimize_aco(
    active_goods: Mapping[int, float],
    drones: Sequence[DroneState],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    aco_config: Mapping,
    seed: int,
    central_node: int = 0,
    objective_config: Mapping | None = None,
) -> RoutingSolution:
    """Visit as many filtered warehouses as feasible, then minimise route distance."""
    if not active_goods:
        return idle_solution(drones, active_goods, central_node)
    if len(drones) != 1:
        raise ValueError("Single-drone ACO requires exactly one configured drone.")

    objective = aco_config if objective_config is None else objective_config
    rng = np.random.default_rng(seed)
    pheromone = np.full(distance_matrix.shape, float(aco_config["initial_pheromone"]), dtype=float)
    np.fill_diagonal(pheromone, 0.0)
    global_best: RoutingSolution | None = None
    stalled = 0

    for _ in range(int(aco_config["iterations"])):
        iteration_best: RoutingSolution | None = None
        for _ in range(int(aco_config["ants"])):
            candidate = _construct_ant_solution(
                active_goods=active_goods,
                drones=drones,
                distance_matrix=distance_matrix,
                node_to_index=node_to_index,
                pheromone=pheromone,
                rng=rng,
                alpha=float(aco_config["alpha"]),
                beta=float(aco_config["beta"]),
                unvisited_penalty=float(objective["unvisited_penalty"]),
                distance_weight=float(objective["distance_weight"]),
                central_node=central_node,
            )
            validate_solution(
                candidate,
                drones,
                active_goods,
                distance_matrix,
                node_to_index,
                central_node,
            )
            if iteration_best is None or candidate.objective_value < iteration_best.objective_value:
                iteration_best = candidate

        assert iteration_best is not None
        improved = (
            global_best is None
            or iteration_best.objective_value < global_best.objective_value - 1e-12
        )
        if improved:
            global_best = iteration_best
            stalled = 0
        else:
            stalled += 1

        pheromone *= 1.0 - float(aco_config["evaporation"])
        assert global_best is not None
        deposit = float(aco_config["deposit_scale"]) / (1.0 + global_best.objective_value)
        for route in global_best.routes.values():
            for left, right in zip(route[:-1], route[1:]):
                if left == right:
                    continue
                left_index, right_index = node_to_index[left], node_to_index[right]
                pheromone[left_index, right_index] += deposit
                pheromone[right_index, left_index] += deposit
        pheromone = np.clip(
            pheromone,
            float(aco_config["pheromone_min"]),
            float(aco_config["pheromone_max"]),
        )
        np.fill_diagonal(pheromone, 0.0)
        if stalled >= int(aco_config["stall_iterations"]):
            break

    assert global_best is not None
    validate_solution(
        global_best, drones, active_goods, distance_matrix, node_to_index, central_node
    )
    return global_best
