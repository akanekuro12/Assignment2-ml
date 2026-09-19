from __future__ import annotations

from copy import deepcopy
import unittest

import pandas as pd

from src.config_loader import load_config
from src.rolling_simulation import run_simulation


class RollingSimulationTests(unittest.TestCase):
    def test_queue_is_reduced_only_by_drone_pickup(self) -> None:
        config = deepcopy(load_config())
        config["data"]["warehouse_ids"] = [1]
        config["warehouses"]["capacities"] = {1: 10}
        config["warehouses"]["coordinates_km"] = {0: [0, 0], 1: [1, 0]}
        config["drones"]["fleet"] = [
            {
                "id": 1,
                "payload_capacity": 5,
                "battery_max": 10,
                "energy_per_km": 1,
                "max_route_distance_km": 10,
            }
        ]
        intervals = pd.DataFrame(
            {
                "time_bin": [0],
                "dock": [1],
                "arrivals": [9],
                "mean_arrivals_60m": [9.0],
                "interval_start_seconds": [0],
                "day": [1],
                "hour": [0],
            }
        )
        log, _ = run_simulation(intervals, config, "fullest_first", seed=42)
        self.assertEqual(log.loc[0, "amount_picked"], 5)
        self.assertEqual(log.loc[0, "queue_after"], 4)
        self.assertEqual(log.loc[0, "overflow"], 0)

    def test_drone_battery_is_restored_before_each_interval(self) -> None:
        config = deepcopy(load_config())
        config["data"]["warehouse_ids"] = [1]
        config["warehouses"]["capacities"] = {1: 10}
        config["warehouses"]["coordinates_km"] = {0: [0, 0], 1: [1, 0]}
        config["drones"]["fleet"] = [
            {
                "id": 1,
                "payload_capacity": 5,
                "battery_max": 2,
                "energy_per_km": 1,
                "max_route_distance_km": 2,
            }
        ]
        intervals = pd.DataFrame(
            {
                "time_bin": [0, 1],
                "dock": [1, 1],
                "arrivals": [9, 9],
                "mean_arrivals_60m": [9.0, 9.0],
                "interval_start_seconds": [0, 900],
                "day": [1, 1],
                "hour": [0, 0],
            }
        )

        log, routes = run_simulation(intervals, config, "fullest_first", seed=42)

        self.assertEqual(log["amount_picked"].tolist(), [5.0, 5.0])
        self.assertEqual(routes["energy_used"].tolist(), [2.0, 2.0])
        self.assertEqual(routes["battery_remaining"].tolist(), [0.0, 0.0])

    def test_mlp_aco_never_routes_a_warehouse_below_70_percent(self) -> None:
        class Predictor:
            def predict_by_warehouse(self, features: pd.DataFrame) -> dict[int, float]:
                return {1: 0.699, 2: 0.70}

        config = deepcopy(load_config())
        config["data"]["warehouse_ids"] = [1, 2]
        config["warehouses"]["capacities"] = {1: 10, 2: 10}
        config["warehouses"]["coordinates_km"] = {
            0: [0, 0], 1: [1, 0], 2: [0, 1]
        }
        intervals = pd.DataFrame(
            {
                "time_bin": [0, 0],
                "dock": [1, 2],
                "arrivals": [5, 5],
                "mean_arrivals_60m": [5.0, 5.0],
                "interval_start_seconds": [0, 0],
                "day": [1, 1],
                "hour": [0, 0],
            }
        )

        log, routes = run_simulation(
            intervals, config, "mlp_aco", predictor=Predictor(), seed=42
        )

        self.assertEqual(log.set_index("dock")["active"].to_dict(), {1: False, 2: True})
        self.assertEqual(routes.loc[0, "route"], "[0, 2, 0]")


if __name__ == "__main__":
    unittest.main()
