# WEPAStacks: MLP + ACO cho một drone

Dự án tách thành hai bài toán độc lập:

1. **MLP dự đoán:** với từng kho, trả về xác suất kho sẽ đạt mức đầy 80% trong 60 phút tới.
2. **ACO định tuyến:** chỉ nhận các kho có xác suất `>= 0.70`, rồi tìm một route ngắn cho đúng một drone dưới các giới hạn vật lý.

Không có `urgency`, `projected_stock` hoặc `required_pickup` trong ACO. Các kho có xác suất dưới 70% bị loại trước khi ACO chạy và không thể xuất hiện trong route.

## 1. Dữ liệu và MLP

Mỗi observation là trạng thái một kho ở cuối interval 15 phút. Vector input có bảy phần tử:

```text
[buffer_fill_ratio,
 arrivals_15m,
 mean_arrivals_60m,
 dock_1, dock_2, dock_3, dock_4]
```

Target là nhãn nhị phân `congestion = 1` nếu lượng hàng trong kịch bản không có drone đạt ít nhất 80% capacity trong bốn interval tiếp theo. Dữ liệu tương lai chỉ tạo label, không đi vào input.

```text
Input(7) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Sigmoid)
```

Output sigmoid là xác suất trong `[0, 1]`, không phải nhãn binary. Ngưỡng vận hành cố định là `0.70`:

```text
active_warehouses = {i | probability[i] >= 0.70 and available[i] > 0}
```

Dữ liệu được chia theo thời gian: ngày 1–60 train, ngày 61–74 validation, từ ngày 75 trở đi test. Scaler chỉ fit trên train.

## 2. ACO một drone

Input của ACO gồm danh sách kho đã qua bộ lọc 70% và lượng hàng hiện có, ma trận khoảng cách, một drone, pheromone và các tham số ACO.

Mỗi ant xây một route bắt đầu tại depot `0`. Ở mỗi bước, ant chỉ xét kho chưa ghé mà drone vẫn đủ pin và đủ giới hạn quãng đường để đi tới kho đó rồi quay về depot. Xác suất chọn cạnh chỉ dựa trên pheromone và khoảng cách:

```text
score(i, j) = pheromone(i, j)^alpha * (1 / distance(i, j))^beta
```

Objective:

```text
objective = 1000 * number_of_unvisited_active_warehouses
          + 1 * route_distance
```

Vì route bị giới hạn tối đa 30 km, penalty 1000 khiến thuật toán ưu tiên ghé nhiều kho active nhất; nếu số kho ghé bằng nhau, route ngắn hơn thắng. Kho active không thể ghé trong interval hiện tại vẫn nằm trong queue để xét lại ở interval sau.

Ràng buộc:

```text
route starts and ends at depot 0
route_distance <= max_route_distance_km
route_distance * energy_per_km <= available_battery
sum(pickup) <= payload_capacity
pickup[i] <= available[i]
```

Tải trọng được chia theo tỷ lệ lượng hàng giữa các kho trên route. Pin được reset tại đầu mỗi interval, tương ứng giả định thay/sạc pin ở depot.

## 3. Luồng chạy

```text
WEPAStack events
  -> aggregate theo 15 phút
  -> trạng thái queue của từng kho
  -> MLP probability
  -> lọc probability >= 0.70
  -> single-drone ACO
  -> kiểm tra pin, quãng đường và payload
  -> cập nhật pickup, queue và overflow
```

## 4. Chạy project

```bash
source .venv/bin/activate
python run_training.py
python run_experiment.py --max-intervals 20 --aco-seeds 1
```

Chạy test:

```bash
PYTHONPYCACHEPREFIX=/tmp/assignment2_pycache \
.venv/bin/python -m unittest discover -s tests -v
```

Model hiện tại nằm trong `model_outputs_single_drone/`; kết quả simulation nằm trong
`outputs_single_drone/`. Các artifact multi-drone cũ đã được loại bỏ.

Các bảng và biểu đồ chính được tạo tự động khi train/chạy experiment:

```text
model_outputs_single_drone/classification_metrics_at_070.csv
model_outputs_single_drone/per_warehouse_classification_metrics.csv
model_outputs_single_drone/prediction_baseline_comparison.csv
model_outputs_single_drone/figures/
outputs_single_drone/tables/key_results.csv
outputs_single_drone/tables/constraint_summary.csv
outputs_single_drone/figures/
```

Bản tóm tắt kết quả cuối cùng nằm tại `outputs_single_drone/RESULTS.md`.

## 5. File chính

```text
config/experiment.yaml       threshold, drone và ACO config
src/feature_builder.py       feature và congestion label
src/train_mlp_aco.py         train MLP
src/predict_risk.py          trả xác suất theo từng kho
src/routing_utils.py         lọc 70%, pickup và objective
src/aco_optimizer.py         single-drone ACO
src/solution_validator.py    kiểm tra constraint độc lập
src/rolling_simulation.py    pipeline end-to-end
```

Đặc tả triển khai chi tiết bằng tiếng Việt nằm tại `docs/implementation_guide_single_drone_mlp_aco_vi.md`.

## 6. Giới hạn

- Congestion label được tạo bằng simulation, không phải nhãn congestion quan sát trực tiếp.
- Capacity, tọa độ, depot và thông số drone là giả định thí nghiệm.
- Một event được xem là một standardized cargo unit.
- Chưa mô phỏng thời gian bay, thời gian sạc, thời tiết, vùng cấm bay hay tránh va chạm.
- Với chỉ bốn kho, exact solver được giữ để đối chiếu ACO trên instance nhỏ.
