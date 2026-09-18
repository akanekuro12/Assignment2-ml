# WEPAStacks MLP–ACO/PSO Drone Simulation

Dự án sử dụng luồng pallet inbound thật của WEPAStacks để tạo nhu cầu theo thời gian. Bốn inbound point được ánh xạ thành bốn kho vệ tinh **giả lập**; vị trí kho, sức chứa, depot trung tâm và đội drone đều là giả định thí nghiệm.

Notebook MLP cũ ở `src/MLP_inbound_congestion_training.ipynb` và artifacts trong `model_outputs/` được giữ nguyên để làm baseline. Pipeline mới không dùng `service_rate`: hàng chỉ rời queue khi được drone lấy.

## 1. Task definition và system interface

### Training input

Mỗi observation biểu diễn một kho tại cuối một interval 15 phút. Vector đầu vào có đúng bảy phần tử theo thứ tự:

```text
[buffer_fill_ratio,
 arrivals_15m,
 mean_arrivals_60m,
 dock_1, dock_2, dock_3, dock_4]
```

- `buffer_fill_ratio = available_before_pickup / warehouse_capacity`.
- `arrivals_15m` là số cargo unit đến trong interval hiện tại.
- `mean_arrivals_60m` là trung bình interval hiện tại và ba interval trước.
- Bốn trường `dock_*` là one-hot encoding.
- Chỉ thông tin tại hoặc trước thời điểm `t` được dùng làm input.

### Training output

Output target là biến nhị phân:

```text
congestion = 1
```

nếu, trong giả định không có drone ở bốn interval tiếp theo, lượng hàng đạt ít nhất 80% sức chứa kho. Future arrivals chỉ được dùng để tạo label lịch sử, không được đưa vào feature.

### Deployment input và output

Trong rolling simulation, hệ thống nhận trạng thái hiện tại của bốn kho và trả về:

1. MLP trả `risk_probability[i, t]` trong `[0, 1]` cho mỗi kho.
2. ACO hoặc PSO trả route của mỗi drone, lượng pickup tại mỗi kho, năng lượng, hàng còn lại và objective value.
3. Mỗi route bắt đầu và kết thúc tại depot `0`.

## 2. Data flow

```text
CSV events
  -> clean and aggregate into 15-minute intervals
  -> full time-bin x warehouse grid
  -> fixed fullest-first historical simulation
  -> leakage-safe MLP features and labels
  -> chronological train / validation / test
  -> saved Keras model + scaler + configuration

Held-out arrivals
  -> current queue + arrivals
  -> MLP risk probability
  -> active warehouse decision
  -> ACO or PSO routes and pickups
  -> constraint validation
  -> queue, overflow and battery update
  -> operational metrics
```

## 3. Theory-to-code mapping

| Khái niệm | Công thức hoặc hành vi | Code |
|---|---|---|
| Full grid | Mọi time bin có đủ bốn kho | `src/data_pipeline.py` |
| Queue | `available = Q + arrivals` | `src/queue_simulator.py`, `src/rolling_simulation.py` |
| Queue update | `Q_next = min(capacity, max(0, available - pickup))` | `src/rolling_simulation.py` |
| Overflow | `max(0, available - pickup - capacity)` | `src/rolling_simulation.py` |
| MLP | Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, sigmoid) | `src/train_mlp_aco.py` |
| Loss | Weighted binary cross-entropy | `src/train_mlp_aco.py` |
| ACO transition | `tau(i,j)^alpha * eta(i,j)^beta` | `src/aco_optimizer.py` |
| Heuristic | `(epsilon + urgency) / (epsilon + distance)` | `src/aco_optimizer.py` |
| PSO update | `v = wv + c1*r1*(pbest-x) + c2*r2*(gbest-x)` | `src/pso_optimizer.py` |
| PSO decoder | Random keys → drone order, visit limit và warehouse order | `src/pso_optimizer.py` |
| Return reserve | Pin phải đủ đi tới node và quay về depot | `src/aco_optimizer.py` |
| Feasibility | Payload, battery, visited pickup, demand | `src/solution_validator.py` |

ACO và PSO tối thiểu hóa cùng một objective:

```text
objective = weight_energy * normalized_energy
          + weight_remaining * normalized_risk_weighted_remaining
```

Loss của MLP và objective vận hành không giống nhau. Binary cross-entropy giúp học xác suất congestion, trong khi mục tiêu thực tế còn quan tâm overflow, hàng thu gom và năng lượng. Vì vậy dự án đánh giá cả prediction metrics lẫn operational metrics.

## 4. Chạy chương trình

Kích hoạt môi trường:

```bash
source .venv/bin/activate
```

Huấn luyện model tương thích drone-only:

```bash
python run_training.py
```

Artifacts mới được lưu trong `model_outputs_aco/`. File `.keras` là model đã huấn luyện; notebook không thay thế trực tiếp file này.

Chạy một smoke experiment ngắn:

```bash
python run_experiment.py --max-intervals 20 --optimizer-seeds 1
```

Chạy thí nghiệm đầy đủ theo các policy và 10 optimizer seeds trong config:

```bash
python run_experiment.py
```

Thí nghiệm đầy đủ có thể mất nhiều thời gian vì ACO được chạy lại ở từng interval.

Chạy unit tests mà không cần cài pytest:

```bash
PYTHONPYCACHEPREFIX=/tmp/assignment2_pycache \
python -m unittest discover -s tests -v
```

## 5. Cấu trúc chính

```text
config/experiment.yaml       cấu hình duy nhất của thí nghiệm
src/data_pipeline.py         đọc, làm sạch và aggregate dữ liệu
src/queue_simulator.py       fixed policy tạo historical training states
src/feature_builder.py       features, labels và chronological splits
src/train_mlp_aco.py         train và lưu MLP artifacts
src/predict_risk.py          deployment interface của MLP
src/aco_optimizer.py         multi-drone ACO
src/pso_optimizer.py         random-key multi-drone PSO
src/baselines.py             nearest-first và fullest-first
src/exact_solver.py          exhaustive reference cho tối đa bốn kho
src/rolling_simulation.py    end-to-end operational simulation
src/solution_validator.py    independent constraint checks
src/metrics.py               operational metrics và multi-seed summary
```

## 6. Giới hạn cần trình bày trong report

- Congestion target được tạo bằng simulation, không phải nhãn congestion thật.
- Capacity, khoảng cách và thông số drone là giả định.
- Một event được gọi là `standardized cargo unit`, không khẳng định drone chở nguyên pallet thật.
- Dock 1 chiếm phần lớn sự kiện; validation/test có thể không có positive label ở Dock 2–4.
- Probability của MLP có thể cần calibration.
- Với chỉ bốn kho, exact reference cần được dùng để kiểm tra ACO và PSO trên các instance nhỏ.
- ACO phù hợp tự nhiên với route rời rạc; PSO cần random-key decoder nên kết quả phụ thuộc cả swarm update lẫn cách decode.
