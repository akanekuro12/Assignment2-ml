from __future__ import annotations

from copy import deepcopy
import unittest

from src.config_loader import load_config
from src.part_b_experiments import build_synthetic_routing_scenarios


class PartBExperimentTests(unittest.TestCase):
    def test_synthetic_scenarios_are_reproducible_and_cover_two_to_four_nodes(self) -> None:
        config = load_config()
        config = deepcopy(config)
        config["part_b_experiments"]["synthetic_cases_per_size"] = 2
        first = build_synthetic_routing_scenarios(config)
        second = build_synthetic_routing_scenarios(config)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(
            [scenario["active_warehouses"] for scenario in first],
            [2, 2, 3, 3, 4, 4],
        )
        for scenario in first:
            self.assertEqual(len(scenario["active_goods"]), scenario["active_warehouses"])
            self.assertEqual(len(scenario["coordinates"]), scenario["active_warehouses"] + 1)
            self.assertGreater(scenario["route_limit"], 0)


if __name__ == "__main__":
    unittest.main()
