# Đặc tả triển khai MLP–ACO một drone

Đây là nguồn hướng dẫn triển khai chính của project và phải được đọc cùng
`config/experiment.yaml`.

## 1. Kiến trúc bắt buộc

Pipeline gồm hai bài toán tách biệt:

1. MLP dự đoán xác suất từng kho sẽ đầy.
2. ACO tìm route cho một drone qua tập kho còn lại sau bộ lọc xác suất.

Không đưa `urgency`, forecast stock, projected demand hoặc required pickup vào ACO.

```text
current warehouse state
        |
        v
MLP probability per warehouse
        |
        v
keep probability >= 0.70
        |
        v
single-drone constrained ACO
        |
        v
route + proportional pickup
```

## 2. Cấu hình

```yaml
mlp:
  horizon_intervals: 4
  congestion_threshold: 0.80
  risk_threshold: 0.70

drones:
  fleet:
    - id: 1
      payload_capacity: 30
      battery_max: 30.0
      energy_per_km: 1.0
      max_route_distance_km: 30.0

objective:
  unvisited_penalty: 1000.0
  distance_weight: 1.0

simulation:
  central_node: 0
  battery_reset_policy: per_interval
```

Config loader phải từ chối cấu hình có số drone khác một, thông số drone không dương, hoặc battery policy khác `per_interval`.

## 3. MLP

### Input

```text
[buffer_fill_ratio, arrivals_15m, mean_arrivals_60m,
 dock_1, dock_2, dock_3, dock_4]
```

### Target và output

Target binary bằng 1 nếu no-drone stock đạt ít nhất 80% capacity trong 60 phút tiếp theo. Output sigmoid là xác suất liên tục:

```text
p_i = P(congestion_i = 1 | current features), 0 <= p_i <= 1
```

Kiến trúc:

```text
Input(7) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Sigmoid)
```

Chia dữ liệu chronological: ngày 1–60 train, 61–74 validation, 75+ test. Fit scaler trên train và chỉ transform validation/test. Future arrivals chỉ tạo label, tuyệt đối không đi vào features.

Validation-optimal threshold có thể được lưu như một chỉ số chẩn đoán, nhưng không điều khiển route. Ngưỡng vận hành của bài toán này luôn lấy từ `mlp.risk_threshold = 0.70`.

## 4. Bộ lọc trước ACO

Tại interval `t`:

```text
available[i] = queue[i] + arrivals[i]

active[i] = (
    available[i] > 0
    and risk_probability[i] >= 0.70
)
```

Kho không active biến mất khỏi input của ACO. Validator phải báo lỗi nếu route chứa một kho đã bị lọc.

## 5. ACO

### Input

```text
active_goods: {warehouse_id: available_goods}
one DroneState
distance_matrix
node_to_index
ACO hyperparameters
```

MLP probability không được dùng lại trong heuristic hoặc objective.

### Xây route của một ant

```text
route = [depot]
current = depot
distance_so_far = 0
unvisited = set(active warehouses)

while unvisited is not empty:
    feasible = []

    for warehouse j in unvisited:
        projected_distance = (
            distance_so_far
            + distance(current, j)
            + distance(j, depot)
        )
        projected_energy = projected_distance * energy_per_km

        if projected_distance <= max_route_distance_km
           and projected_energy <= available_battery:
            feasible.append(j)

    if feasible is empty:
        break

    for j in feasible:
        heuristic[j] = 1 / (epsilon + distance(current, j))
        score[j] = pheromone[current, j]^alpha * heuristic[j]^beta

    next = random_choice(feasible, normalize(score))
    append next to route
    update distance_so_far, current and unvisited

append depot to route
```

### Objective

```text
unvisited_count = active warehouse count - visited active warehouse count
objective = 1000 * unvisited_count + route_distance
```

Với giới hạn route 30 km, objective này tương đương:

1. tối đa hóa số kho active được ghé;
2. trong các route ghé cùng số kho, chọn route ngắn nhất.

Pheromone bay hơi sau mỗi iteration; các cạnh của global-best route được tăng pheromone. Thuật toán dừng khi đạt số iteration tối đa hoặc không cải thiện đủ lâu.

## 6. Pickup và constraint

Routing không dùng lượng hàng để tạo độ ưu tiên. Sau khi có route, payload được chia theo tỷ lệ hàng hiện có tại các kho đã ghé:

```text
route_available = sum(available[j] for j in visited)
payload_used = min(payload_capacity, route_available)
pickup[j] = payload_used * available[j] / route_available
```

Phải kiểm tra độc lập:

```text
route[0] == depot and route[-1] == depot
each active warehouse appears at most once
no filtered warehouse appears in route
route_distance <= max_route_distance_km
route_distance * energy_per_km <= available_battery
sum(pickup) <= payload_capacity
0 <= pickup[j] <= available[j]
```

Kho active chưa ghé hoặc hàng chưa lấy hết vẫn nằm trong queue và được xét lại ở interval sau.

## 7. Queue update

```text
raw_next[i] = max(0, available[i] - pickup[i])
overflow[i] = max(0, raw_next[i] - capacity[i])
queue_next[i] = min(capacity[i], raw_next[i])
```

Pin bị trừ sau route để ghi log. Đầu interval kế tiếp, pin reset về `battery_max`; đây là giả định thay hoặc sạc pin tại depot.

## 8. Output bắt buộc

Interval log:

```text
available_before_pickup
risk_probability
active
visited
amount_picked
queue_after
overflow
constraint_violations
```

Route log:

```text
route
pickup
route_distance
energy_used
battery_remaining
active_warehouse_count
visited_warehouse_count
unvisited_warehouse_count
objective_value
```

## 9. Test bắt buộc

1. Xác suất `0.70` được giữ; `0.699` bị loại.
2. Kho không có hàng không đi vào ACO dù xác suất cao.
3. Route chỉ chứa kho đã qua bộ lọc.
4. Route bắt đầu và kết thúc ở depot.
5. Không lặp kho.
6. Tổng pickup không vượt payload và pickup không vượt hàng hiện có.
7. Route thỏa cả battery và `max_route_distance_km`.
8. Cùng seed tạo cùng kết quả ACO.
9. Pin reset mỗi interval.
10. Exact solver đối chiếu ACO trên tối đa bốn kho.

## 10. Lệnh xác minh

```bash
PYTHONPYCACHEPREFIX=/tmp/assignment2_pycache \
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python run_training.py
.venv/bin/python run_experiment.py --max-intervals 20 --aco-seeds 1
```

Khi chạy lại pipeline, artifact MLP sẽ được tạo trong
`model_outputs_single_drone/` và kết quả routing sẽ được tạo trong
`outputs_single_drone/`. Hai thư mục này hiện không phải source code và có thể
được tạo lại từ dữ liệu, cấu hình và các entry point của project.
