"""Shared domain objects for routing and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np


@dataclass
class DroneState:
    id: int
    payload_capacity: float
    battery_max: float
    energy_per_km: float
    max_route_distance_km: float | None = None
    available_battery: float | None = None

    def __post_init__(self) -> None:
        if self.available_battery is None:
            self.available_battery = float(self.battery_max)
        if self.max_route_distance_km is None:
            self.max_route_distance_km = float("inf")

    def reset_battery(self) -> None:
        self.available_battery = float(self.battery_max)

    def copy(self) -> "DroneState":
        return DroneState(
            id=self.id,
            payload_capacity=self.payload_capacity,
            battery_max=self.battery_max,
            energy_per_km=self.energy_per_km,
            max_route_distance_km=self.max_route_distance_km,
            available_battery=self.available_battery,
        )


@dataclass
class RoutingSolution:
    routes: Dict[int, List[int]]
    pickups: Dict[int, Dict[int, float]]
    route_distance: Dict[int, float]
    energy_used: Dict[int, float]
    remaining_goods: Dict[int, float]
    objective_value: float
    feasible: bool = True
    violations: List[str] = field(default_factory=list)

    @property
    def total_energy(self) -> float:
        return float(sum(self.energy_used.values()))

    @property
    def total_pickup(self) -> float:
        return float(sum(sum(values.values()) for values in self.pickups.values()))


def drones_from_config(config: Mapping) -> list[DroneState]:
    return [
        DroneState(
            id=int(item["id"]),
            payload_capacity=float(item["payload_capacity"]),
            battery_max=float(item["battery_max"]),
            energy_per_km=float(item["energy_per_km"]),
            max_route_distance_km=float(item["max_route_distance_km"]),
        )
        for item in config["drones"]["fleet"]
    ]


def euclidean_distance_matrix(
    coordinates: Mapping[int, tuple[float, float]],
) -> tuple[np.ndarray, dict[int, int], list[int]]:
    nodes = sorted(int(node) for node in coordinates)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for left in nodes:
        for right in nodes:
            x1, y1 = coordinates[left]
            x2, y2 = coordinates[right]
            matrix[node_to_index[left], node_to_index[right]] = float(
                np.hypot(x1 - x2, y1 - y2)
            )
    return matrix, node_to_index, nodes


def route_distance(
    route: list[int], distance_matrix: np.ndarray, node_to_index: Mapping[int, int]
) -> float:
    return float(
        sum(
            distance_matrix[node_to_index[left], node_to_index[right]]
            for left, right in zip(route[:-1], route[1:])
        )
    )
