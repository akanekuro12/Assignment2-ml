"""Independent feasibility checks for every routing solution."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .domain import DroneState, RoutingSolution, route_distance


def validate_solution(
    solution: RoutingSolution,
    drones: Sequence[DroneState],
    demand: Mapping[int, float],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    central_node: int = 0,
    tolerance: float = 1e-8,
    raise_on_error: bool = True,
) -> list[str]:
    """Recompute constraints instead of trusting cached solution values."""
    errors: list[str] = []
    valid_nodes = set(node_to_index)
    drone_by_id = {drone.id: drone for drone in drones}
    pickup_by_warehouse = {int(node): 0.0 for node in demand}

    for drone_id, drone in drone_by_id.items():
        route = solution.routes.get(drone_id, [])
        pickups = solution.pickups.get(drone_id, {})
        if len(route) < 2 or route[0] != central_node or route[-1] != central_node:
            errors.append(f"Drone {drone_id}: route must start and end at {central_node}.")
        if any(node not in valid_nodes for node in route):
            errors.append(f"Drone {drone_id}: route contains an unknown node.")

        total_pickup = float(sum(pickups.values()))
        if total_pickup > drone.payload_capacity + tolerance:
            errors.append(f"Drone {drone_id}: payload capacity exceeded.")
        if any(float(amount) < -tolerance for amount in pickups.values()):
            errors.append(f"Drone {drone_id}: pickup cannot be negative.")
        for warehouse, amount in pickups.items():
            if float(amount) > tolerance and int(warehouse) not in route:
                errors.append(f"Drone {drone_id}: pickup at unvisited warehouse {warehouse}.")
            pickup_by_warehouse[int(warehouse)] = pickup_by_warehouse.get(int(warehouse), 0.0) + float(amount)

        if route and all(node in valid_nodes for node in route):
            computed_distance = route_distance(route, distance_matrix, node_to_index)
            computed_energy = computed_distance * drone.energy_per_km
            if computed_energy > float(drone.available_battery) + tolerance:
                errors.append(f"Drone {drone_id}: available battery exceeded.")
            if abs(computed_energy - float(solution.energy_used.get(drone_id, 0.0))) > tolerance:
                errors.append(f"Drone {drone_id}: cached energy does not match its route.")

    for warehouse, amount in pickup_by_warehouse.items():
        if amount > float(demand.get(warehouse, 0.0)) + tolerance:
            errors.append(f"Warehouse {warehouse}: pickup exceeds demand.")

    solution.feasible = not errors
    solution.violations = errors
    if errors and raise_on_error:
        raise AssertionError("Invalid routing solution:\n- " + "\n- ".join(errors))
    return errors


def unreachable_round_trips(
    drones: Sequence[DroneState],
    warehouses: Sequence[int],
    distance_matrix: np.ndarray,
    node_to_index: Mapping[int, int],
    central_node: int = 0,
) -> list[str]:
    warnings: list[str] = []
    for warehouse in warehouses:
        feasible_drone_exists = False
        distance = 2.0 * distance_matrix[node_to_index[central_node], node_to_index[warehouse]]
        for drone in drones:
            if distance * drone.energy_per_km <= drone.battery_max + 1e-8:
                feasible_drone_exists = True
                break
        if not feasible_drone_exists:
            warnings.append(
                f"Warehouse {warehouse} cannot be served by any drone with a return reserve."
            )
    return warnings
