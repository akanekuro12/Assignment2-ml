"""Run operational baselines, ACO and PSO on the held-out chronological period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config_loader import DEFAULT_CONFIG_PATH, load_config, resolve_project_path
from src.data_pipeline import load_interval_table
from src.metrics import operational_metrics, summarize_runs
from src.predict_risk import RiskPredictor
from src.rolling_simulation import SUPPORTED_POLICIES, run_simulation


def _parse_policies(value: str) -> list[str]:
    policies = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(policies).difference(SUPPORTED_POLICIES)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown policies: {sorted(unknown)}")
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--policies",
        type=_parse_policies,
        default=None,
        help="Comma-separated policies; default comes from experiment.yaml.",
    )
    parser.add_argument("--max-intervals", type=int, default=None)
    parser.add_argument(
        "--optimizer-seeds", "--aco-seeds", dest="optimizer_seeds", type=int,
        default=None, help="Limit the configured stochastic-optimizer seed list.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    intervals = load_interval_table(config)
    test_start_day = int(config["mlp"]["validation_end_day"]) + 1
    intervals = intervals.loc[intervals["day"].ge(test_start_day)].copy()
    if args.max_intervals is not None:
        selected_bins = sorted(intervals["time_bin"].unique())[: args.max_intervals]
        intervals = intervals.loc[intervals["time_bin"].isin(selected_bins)].copy()
    intervals = intervals.dropna(subset=["mean_arrivals_60m"])
    policies = args.policies or list(config["simulation"]["test_policies"])

    predictor = None
    mlp_policies = {"mlp_aco", "mlp_pso"}
    if mlp_policies.intersection(policies):
        model_directory = resolve_project_path(config["outputs"]["model_directory"], config)
        if not (model_directory / "mlp_congestion_aco.keras").exists():
            raise FileNotFoundError(
                f"ACO-compatible model not found in {model_directory}. Run run_training.py first."
            )
        predictor = RiskPredictor(model_directory)

    output_root = resolve_project_path(config["outputs"]["experiment_directory"], config)
    log_directory = output_root / "logs"
    route_directory = output_root / "routes"
    table_directory = output_root / "tables"
    for directory in (log_directory, route_directory, table_directory):
        directory.mkdir(parents=True, exist_ok=True)

    seed_values = config["simulation"].get(
        "optimizer_seeds", config["simulation"].get("aco_seeds", [config["seed"]])
    )
    configured_seeds = [int(value) for value in seed_values]
    if args.optimizer_seeds is not None:
        configured_seeds = configured_seeds[: args.optimizer_seeds]
    metric_rows = []
    for policy in policies:
        stochastic = {"aco_current", "mlp_aco", "pso_current", "mlp_pso"}
        seeds = configured_seeds if policy in stochastic else [int(config["seed"])]
        for seed in seeds:
            interval_log, route_log = run_simulation(
                intervals,
                config,
                policy,
                predictor=predictor if policy in mlp_policies else None,
                seed=seed,
            )
            suffix = f"{policy}_seed_{seed}"
            interval_log.to_csv(log_directory / f"interval_{suffix}.csv", index=False)
            route_log.to_csv(route_directory / f"routes_{suffix}.csv", index=False)
            metrics = operational_metrics(interval_log)
            metric_rows.append({"policy": policy, "seed": seed, **metrics})
            print(json.dumps({"policy": policy, "seed": seed, **metrics}, indent=2))

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(table_directory / "operational_metrics_by_run.csv", index=False)
    summary = summarize_runs(metrics_table)
    summary.to_csv(table_directory / "baseline_comparison.csv")
    print(f"Saved experiment outputs to {output_root}")


if __name__ == "__main__":
    main()
