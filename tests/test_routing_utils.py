from __future__ import annotations

import unittest

import numpy as np

from src.domain import DroneState
from src.routing_utils import allocate_pickups, route_is_feasible, select_active_warehouses

class RoutingUtilityTests(unittest.TestCase):
    def test_mlp_filter_keeps_only_nonempty_nodes_at_or_above_70_percent(self) -> None:
        active = select_active_warehouses(
            available={1: 10.0, 2: 10.0, 3: 0.0, 4: 10.0},
            capacities={1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0},
            risk_probability={1: 0.70, 2: 0.699, 3: 0.99, 4: 0.95},
            risk_threshold=0.70,
            fill_threshold=0.80,
            use_risk=True,
        )
        self.assertEqual(active, [1, 4])

    def test_non_mlp_filter_uses_current_fill_ratio(self) -> None:
        active = select_active_warehouses(
            available={1: 80.0, 2: 79.0},
            capacities={1: 100.0, 2: 100.0},
            risk_probability={1: 0.0, 2: 1.0},
            risk_threshold=0.70,
            fill_threshold=0.80,
            use_risk=False,
        )
        self.assertEqual(active, [1])

    def test_pickup_is_shared_and_never_exceeds_payload(self) -> None:
        drone = DroneState(1, 30.0, 100.0, 1.0)
        pickups, remaining = allocate_pickups(
            [0, 1, 2, 0], {1: 20.0, 2: 40.0}, drone
        )
        self.assertAlmostEqual(sum(pickups.values()), 30.0)
        self.assertAlmostEqual(pickups[1], 10.0)
        self.assertAlmostEqual(pickups[2], 20.0)
        self.assertAlmostEqual(sum(remaining.values()), 30.0)

    def test_route_must_satisfy_battery_and_explicit_distance_limit(self) -> None:
        distances = np.asarray([[0.0, 6.0], [6.0, 0.0]])
        index = {0: 0, 1: 1}
        route = [0, 1, 0]
        distance_limited = DroneState(1, 10.0, 100.0, 1.0, 10.0)
        battery_limited = DroneState(1, 10.0, 10.0, 1.0, 100.0)
        feasible = DroneState(1, 10.0, 12.0, 1.0, 12.0)
        self.assertFalse(route_is_feasible(route, distance_limited, distances, index))
        self.assertFalse(route_is_feasible(route, battery_limited, distances, index))
        self.assertTrue(route_is_feasible(route, feasible, distances, index))


if __name__ == "__main__":
    unittest.main()
