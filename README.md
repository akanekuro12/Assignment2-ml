# WEPAStacks: MLP + ACO for a Single Drone

The project separates two tasks:

1. **MLP prediction:** for each warehouse, estimate the probability that its stock will reach 80% of capacity within the next 60 minutes.
2. **ACO routing:** consider only warehouses with a predicted probability of `>= 0.70`, then find a short route for one drone under physical constraints.

ACO does not use `urgency`, `projected_stock`, or `required_pickup`. Warehouses with a probability below 70% are excluded before ACO runs and cannot appear on its route.

## 1. Data and MLP

Each observation describes one warehouse at the end of a 15-minute interval. The input vector has seven features:

```text
[buffer_fill_ratio,
 arrivals_15m,
 mean_arrivals_60m,
 dock_1, dock_2, dock_3, dock_4]
```

The target is the binary label `congestion = 1` if stock would reach at least 80% of capacity during the next four intervals without a drone pickup. Future data is used only to create the label, never as an input feature.

```text
Input(7) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Sigmoid)
```

The sigmoid output is a probability in `[0, 1]`, not a binary label. The fixed operational threshold is `0.70`:

```text
active_warehouses = {i | probability[i] >= 0.70 and available[i] > 0}
```

Data is split chronologically: days 1–60 for training, days 61–74 for validation, and day 75 onward for testing. The scaler is fitted only on the training set.

## 2. Single-Drone ACO

ACO takes the warehouses that pass the 70% filter and their available stock, a distance matrix, one drone, pheromone values, and ACO parameters.

Each ant builds a route starting at depot `0`. At each step, it considers only unvisited warehouses that the drone can reach while retaining enough battery and route-distance allowance to return to the depot. Edge-selection probabilities depend only on pheromone and distance:

```text
score(i, j) = pheromone(i, j)^alpha * (1 / distance(i, j))^beta
```

Objective:

```text
objective = 1000 * number_of_unvisited_active_warehouses
          + 1 * route_distance
```

Routes are limited to 30 km, so the penalty of 1000 makes the algorithm prioritize visiting as many active warehouses as possible. If two routes visit the same number of warehouses, the shorter route wins. An active warehouse that cannot be visited in the current interval remains in the queue for consideration in the next interval.

Constraints:

```text
route starts and ends at depot 0
route_distance <= max_route_distance_km
route_distance * energy_per_km <= available_battery
sum(pickup) <= payload_capacity
pickup[i] <= available[i]
```

The payload is allocated among warehouses on the route in proportion to their available stock. The battery is reset at the start of each interval, representing battery replacement or charging at the depot.

## 3. Execution Flow

```text
WEPAStacks events
  -> aggregate into 15-minute intervals
  -> queue state for each warehouse
  -> MLP probability
  -> filter for probability >= 0.70
  -> single-drone ACO
  -> check battery, distance, and payload limits
  -> update pickups, queues, and overflow
```

## 4. Running the Project

```bash
source .venv/bin/activate
python run_training.py
python run_experiment.py --threshold-sweep
python run_part_b_experiments.py
```

Run the tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/assignment2_pycache \
.venv/bin/python -m unittest discover -s tests -v
```

After running the commands, the trained model and architecture/optimizer validation artifacts are stored in `model_outputs_single_drone/`. Simulation results and the synthetic ACO benchmark are stored in `outputs_single_drone/`. The entry points can regenerate these artifacts from the current configuration.

The main tables and figures generated during training and experiments are:

```text
model_outputs_single_drone/classification_metrics_at_070.csv
model_outputs_single_drone/per_warehouse_classification_metrics.csv
model_outputs_single_drone/prediction_baseline_comparison.csv
model_outputs_single_drone/split_classification_metrics.csv
model_outputs_single_drone/validation_threshold_metrics.csv
model_outputs_single_drone/calibration_table.csv
model_outputs_single_drone/multi_seed_metrics.csv
model_outputs_single_drone/ablation_metrics.csv
model_outputs_single_drone/architecture_comparison_summary.csv
model_outputs_single_drone/optimizer_comparison_summary.csv
model_outputs_single_drone/figures/
outputs_single_drone/tables/key_results.csv
outputs_single_drone/tables/constraint_summary.csv
outputs_single_drone/tables/operational_threshold_sensitivity.csv
outputs_single_drone/tables/aco_exact_benchmark_summary.csv
outputs_single_drone/figures/
```

Do not treat files in either output directory as source files or final evidence without recording the run scope, seed, and configuration that produced them.

## 5. Main Files

```text
config/experiment.yaml       thresholds, drone settings, and ACO configuration
src/feature_builder.py       features and congestion labels
src/train_mlp_aco.py         MLP training
src/predict_risk.py          per-warehouse risk probabilities
src/routing_utils.py         70% filter, pickups, and objective
src/aco_optimizer.py         single-drone ACO
src/solution_validator.py    independent constraint checks
src/rolling_simulation.py    end-to-end pipeline
src/part_b_experiments.py    architecture, optimizer, and synthetic ACO checks
run_part_b_experiments.py    entry point for Part B experiments
```

A detailed implementation guide in Vietnamese is available at `docs/implementation_guide_single_drone_mlp_aco_vi.md`.

## 6. Limitations

- Congestion labels are generated by simulation rather than observed congestion outcomes.
- Warehouse capacities, coordinates, the depot, and drone specifications are experimental assumptions.
- Each event is treated as one standardized cargo unit.
- Flight time, charging time, weather, no-fly zones, and collision avoidance are not simulated.
- With only four warehouses, the exact solver is retained as a reference for comparing ACO on small instances.
