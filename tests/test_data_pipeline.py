from __future__ import annotations

import unittest

import pandas as pd

from src.data_pipeline import aggregate_intervals, clean_events


class DataPipelineTests(unittest.TestCase):
    def test_aggregation_adds_zero_rows_and_keeps_duplicate_events(self) -> None:
        raw = pd.DataFrame(
            {
                "arrival_time": [0, 0, 901, 901],
                "dock": [1, 1, 2, 2],
                "type": ["delivery"] * 4,
            }
        )
        clean = clean_events(raw, [1, 2, 3, 4])
        result = aggregate_intervals(clean, [1, 2, 3, 4], interval_seconds=900)
        self.assertEqual(len(result), 8)
        first = result.set_index(["time_bin", "dock"])["arrivals"]
        self.assertEqual(first.loc[(0, 1)], 2)
        self.assertEqual(first.loc[(0, 2)], 0)
        self.assertEqual(first.loc[(1, 2)], 2)
        self.assertTrue(result.groupby("time_bin")["dock"].nunique().eq(4).all())


if __name__ == "__main__":
    unittest.main()
