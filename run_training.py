"""Command-line entry point for the ACO-compatible MLP training pipeline."""

from __future__ import annotations

import argparse
import json

from src.config_loader import DEFAULT_CONFIG_PATH, load_config
from src.train_mlp_aco import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    metrics = train(load_config(args.config))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
