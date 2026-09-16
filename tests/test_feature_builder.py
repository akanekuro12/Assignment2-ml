from __future__ import annotations

import unittest

import pandas as pd

from src.feature_builder import add_no_drone_labels


class FeatureBuilderTests(unittest.TestCase):
    @staticmethod
    def _state(arrivals: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time_bin": range(len(arrivals)),
                "dock": [1] * len(arrivals),
                "arrivals_15m": arrivals,
                "available_before_pickup": [0.0] * len(arrivals),
            }
        )

    def test_label_boundary_is_inclusive(self) -> None:
        state = self._state([0, 4, 4, 4, 4, 0])
        result = add_no_drone_labels(state, {1: 20}, 4, 0.8)
        self.assertEqual(int(result.loc[0, "congestion"]), 1)

    def test_label_below_boundary_is_negative(self) -> None:
        state = self._state([0, 4, 4, 4, 3.9, 0])
        result = add_no_drone_labels(state, {1: 20}, 4, 0.8)
        self.assertEqual(int(result.loc[0, "congestion"]), 0)

    def test_future_changes_label_but_not_current_features(self) -> None:
        original = self._state([0, 1, 1, 1, 1, 0])
        changed = self._state([0, 10, 10, 10, 10, 0])
        original_label = add_no_drone_labels(original, {1: 20}, 4, 0.8)
        changed_label = add_no_drone_labels(changed, {1: 20}, 4, 0.8)
        self.assertEqual(original.loc[0, "available_before_pickup"], changed.loc[0, "available_before_pickup"])
        self.assertNotEqual(original_label.loc[0, "congestion"], changed_label.loc[0, "congestion"])


if __name__ == "__main__":
    unittest.main()
