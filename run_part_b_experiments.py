"""Run the compact architecture, optimiser, and routing studies used in report Part B."""

from __future__ import annotations

import argparse
import json

from src.config_loader import DEFAULT_CONFIG_PATH, load_config
from src.part_b_experiments import run_part_b_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    results = run_part_b_experiments(load_config(args.config))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
