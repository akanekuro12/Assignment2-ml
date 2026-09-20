"""Run operational baselines and ACO on the held-out chronological period."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_project_path,
    warehouse_coordinates,
)
from src.data_pipeline import load_interval_table
from src.metrics import operational_metrics, summarize_runs
from src.predict_risk import RiskPredictor
from src.result_reporting import (
    save_operational_reports,
    save_operational_threshold_sensitivity,
    save_route_example,
)
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
        "--aco-seeds", type=int, default=None,
        help="Limit the configured ACO seed list.",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Run MLP-ACO over configured risk thresholds using the first ACO seed.",
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
    mlp_policies = {"mlp_aco"}
    model_directory = resolve_project_path(config["outputs"]["model_directory"], config)
    model_path = model_directory / "mlp_congestion_aco.keras"
    if model_path.exists():
        # Predict risk for every policy so risk-weighted metrics are comparable.
        # Only mlp_aco is allowed to use risk when making routing decisions.
        predictor = RiskPredictor(model_directory)
    elif mlp_policies.intersection(policies):
        raise FileNotFoundError(
            f"ACO-compatible model not found in {model_directory}. Run run_training.py first."
        )

    output_root = resolve_project_path(config["outputs"]["experiment_directory"], config)
    log_directory = output_root / "logs"
    route_directory = output_root / "routes"
    table_directory = output_root / "tables"
    for directory in (log_directory, route_directory, table_directory):
        directory.mkdir(parents=True, exist_ok=True)

    seed_values = config["simulation"].get("aco_seeds", [config["seed"]])
    configured_seeds = [int(value) for value in seed_values]
    if args.aco_seeds is not None:
        configured_seeds = configured_seeds[: args.aco_seeds]
    metric_rows = []
    mlp_route_example = None
    for policy in policies:
        stochastic = {"aco_current", "mlp_aco"}
        seeds = configured_seeds if policy in stochastic else [int(config["seed"])]
        for seed in seeds:
            interval_log, route_log = run_simulation(
                intervals,
                config,
                policy,
                predictor=predictor,
                seed=seed,
            )
            suffix = f"{policy}_seed_{seed}"
            interval_log.to_csv(log_directory / f"interval_{suffix}.csv", index=False)
            route_log.to_csv(route_directory / f"routes_{suffix}.csv", index=False)
            metrics = operational_metrics(interval_log)
            metrics["below_threshold_visited"] = (
                int(
                    (
                        interval_log["visited"]
                        & interval_log["risk_probability"].lt(
                            float(config["mlp"]["risk_threshold"])
                        )
                    ).sum()
                )
                if policy == "mlp_aco"
                else 0
            )
            metric_rows.append({"policy": policy, "seed": seed, **metrics})
            if policy == "mlp_aco" and mlp_route_example is None:
                mlp_route_example = route_log.copy()
            print(json.dumps({"policy": policy, "seed": seed, **metrics}, indent=2))

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(table_directory / "operational_metrics_by_run.csv", index=False)
    summary = summarize_runs(metrics_table)
    summary.to_csv(table_directory / "baseline_comparison.csv", index=False)
    summary.to_csv(table_directory / "operational_summary.csv", index=False)
    save_operational_reports(
        run_metrics=metrics_table,
        summary=summary,
        output_root=output_root,
        drone_config=config["drones"]["fleet"][0],
    )
    if mlp_route_example is not None:
        save_route_example(
            mlp_route_example,
            warehouse_coordinates(config),
            output_root / "figures" / "mlp_aco_route_example.png",
        )

    if args.threshold_sweep:
        if predictor is None:
            raise FileNotFoundError("Threshold sensitivity requires a trained MLP model.")
        sweep_seed = configured_seeds[0]
        threshold_rows = []
        configured_threshold = float(config["mlp"]["risk_threshold"])
        thresholds = [float(value) for value in config["simulation"]["threshold_sweep"]]
        for threshold in thresholds:
            existing = metrics_table.loc[
                metrics_table["policy"].eq("mlp_aco")
                & metrics_table["seed"].eq(sweep_seed)
            ]
            if np.isclose(threshold, configured_threshold) and not existing.empty:
                values = existing.iloc[0].drop(labels=["policy", "seed"]).to_dict()
            else:
                sweep_config = deepcopy(config)
                sweep_config["mlp"]["risk_threshold"] = threshold
                interval_log, _ = run_simulation(
                    intervals,
                    sweep_config,
                    "mlp_aco",
                    predictor=predictor,
                    seed=sweep_seed,
                )
                values = operational_metrics(interval_log)
            values.setdefault("below_threshold_visited", 0)
            threshold_rows.append({"threshold": threshold, "seed": sweep_seed, **values})
            print(json.dumps({"threshold": threshold, "seed": sweep_seed, **values}, indent=2))
        save_operational_threshold_sensitivity(pd.DataFrame(threshold_rows), output_root)
    scope = {
        "full_test_period": args.max_intervals is None,
        "time_bins": int(intervals["time_bin"].nunique()),
        "rows": int(len(intervals)),
        "policies": policies,
        "aco_seeds": configured_seeds,
        "prediction_threshold": float(config["mlp"]["risk_threshold"]),
        "threshold_sweep_run": bool(args.threshold_sweep),
    }
    with (output_root / "experiment_scope.json").open("w", encoding="utf-8") as stream:
        json.dump(scope, stream, indent=2, ensure_ascii=False)
    print(f"Saved experiment outputs to {output_root}")


if __name__ == "__main__":
    main()
