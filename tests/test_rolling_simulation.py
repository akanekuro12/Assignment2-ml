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
            {"id": 1, "payload_capacity": 5, "battery_max": 10, "energy_per_km": 1}
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


if __name__ == "__main__":
    unittest.main()
