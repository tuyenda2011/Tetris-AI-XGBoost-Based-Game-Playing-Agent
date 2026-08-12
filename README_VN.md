# Tetris AI: Agent Chơi Game Dựa Trên XGBoost

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/XGBoost-2.0%2B-EE6C4D?logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/License-MIT-00C853" alt="License" />
  <img src="https://img.shields.io/badge/Status-Complete-6C5CE7" alt="Status" />
</p>

<p align="center">
  🌐 <a href="README.md">Tiếng Anh</a> | <b>Tiếng Việt</b>
</p>

Một dự án Học máy Nâng cao huấn luyện mô hình **XGBoost Regressor** để chọn vị trí thả khối Tetris tối ưu dựa trên các đặc trưng hình học của bàn chơi.

---

## 📌 Tổng Quan Dự Án

Khác với Học tăng cường (DQN/PPO) học từ ảnh thô qua cơ chế thử-sai, dự án này sử dụng phương pháp **Học có giám sát trên không gian nước đi ứng viên (Candidate-Action Learning)**:

1. **Sinh vị trí ứng viên**: Đối với mỗi khối gạch xuất hiện, tính toán tất cả các vị trí thả hợp lệ (bao gồm góc xoay và tọa độ cột).
2. **Mô phỏng & Trích xuất đặc trưng**: Mô phỏng thử từng vị trí hạ khối và trích xuất 12 đặc trưng hình học (số lỗ trống, độ gồ ghề, chiều cao, số hàng xóa được...).
3. **Chấm điểm & Quyết định**: Dùng mô hình XGBoost để đánh giá điểm chất lượng của từng vị trí ứng viên và thực thi nước đi có điểm số cao nhất.

```text
Trạng Thái Bàn Chơi & Khối Gạch
               │
    Sinh Các Vị Trí Thả Hợp Lệ (Xoay & Tọa Độ Cột)
               │
Mô Phỏng & Trích Xuất 12 Đặc Trưng Hình Học (Topology Features)
               │
   XGBoost Regressor (Dự Đoán Điểm Chất Lượng Action)
               │
   Chọn & Thực Thi Nước Đi Có Điểm Cao Nhất (Max Score)
```

---

## 🔬 Các Đặc Trưng Bàn Chơi (Engineered Features)

Mô hình đánh giá các nước đi ứng viên dựa trên 12 đặc trưng hình học của bàn chơi sau khi đặt khối:

| Tên Feature | Mô Tả | Ý Nghĩa Trong Game |
| :--- | :--- | :--- |
| `cleared_lines` | Số hàng xóa được ngay lập tức (0–4) | Thưởng ăn điểm chính |
| `aggregate_height` | Tổng chiều cao tất cả 10 cột | Phạt khi bàn chơi dâng quá cao |
| `holes` | Tổng ô trống bị kẹt bên dưới ô đã lấp | Phạt cực nặng (ngăn tạo khoảng hở) |
| `bumpiness` | Độ chênh lệch chiều cao giữa các cột liền kề | Giữ cho bề mặt bàn chơi bằng phẳng |
| `wells` | Độ sâu của các giếng hẹp 1 cột | Tối ưu ô chờ cho khối I (thanh dài) |
| `landing_height` | Độ cao của khối vừa đặt xuống | Khuyến khích hạ khối ở vị trí thấp |
| `max_height` | Cột cao nhất trên bàn chơi | Cảnh báo nguy cơ thua game |
| `min_height` | Cột thấp nhất trên bàn chơi | Đo độ cao nền đáy |
| `height_variance` | Phương sai chiều cao các cột | Đo độ cân bằng tổng thể |
| `occupied_cells` | Tổng số ô gạch đang có trên bàn | Đánh giá độ lấp đầy bàn chơi |
| `row_density` | Mật độ lấp đầy trung bình các hàng | Đo độ đặc của khối gạch theo chiều ngang |
| `col_density` | Mật độ lấp đầy trung bình các cột | Đo độ đặc của khối gạch theo chiều dọc |

---

## ⚙️ Quy Trình Thực Thi Dự Án (Pipeline 5 Bước)

Kích hoạt môi trường Conda trước khi chạy các lệnh:
```bash
conda activate tetris
```

### Bước 1: Kiểm Tra Hệ Thống (Unit Tests)
Kiểm tra logic môi trường game, trích xuất feature và quy tắc sinh nước đi:
```bash
python -m pytest tests/ -v
```

### Bước 2: Sinh Dữ Liệu Huấn Luyện (Dataset Generation)
Sinh tập dữ liệu đa luồng (`ProcessPoolExecutor`) có thanh progress bar hiển thị % thời gian thực:
```bash
python scripts/generate_dataset.py --config configs/dataset_quality.yaml
```

### Bước 3: Huấn Luyện Mô Hình (Train Model)
Train và tự động tinh chỉnh mô hình XGBoost Regressor sử dụng file cấu hình:
```bash
python scripts/train_model.py --config configs/model_train.yaml
```

### Bước 4: Đánh Giá Hiệu Năng (Evaluation)
So sánh hiệu năng của XGBoost Agent với Random Agent baseline:
```bash
python scripts/evaluate_agent.py --episodes 10 --max-pieces 120 --seed 42 --skip-shap
```

### Bước 5: Mở Demo Giao Diện Đồ Họa (Pygame GUI)
Mở cửa sổ đồ họa xem AI chơi game Tetris thời gian thực:
```bash
# XGBoost Model Agent
python scripts/play_gui.py --agent xgboost --seed 42 --delay-ms 150

# Heuristic Agent (Oracle)
python scripts/play_gui.py --agent heuristic --seed 42 --delay-ms 150

# Random Baseline Agent
python scripts/play_gui.py --agent random --seed 42 --delay-ms 150
```

> 🕹️ **Phím tắt GUI:** `UP` / `DOWN` tăng/giảm tốc độ · `SPACE` tạm dừng · `R` chơi lại · `ESC` thoát

---

## 🗂️ Cấu Trúc Thư Mục Dự Án

```text
Tetris-AI-XGBoost-Based-Game-Playing-Agent/
├── configs/                     # File cấu hình YAML
│   └── dataset_quality.yaml     # Tham số sinh dataset
├── src/                         # Mã nguồn cốt lõi
│   ├── environment.py           # Engine Tetris chuẩn Gymnasium API
│   ├── features.py               # Module trích xuất 12 đặc trưng bàn chơi
│   ├── actions.py                # Sinh nước đi ứng viên hợp lệ
│   ├── dataset.py                # Sinh dataset đa luồng (multiprocessing)
│   ├── model.py                  # Module mô hình XGBoost
│   ├── agent.py                  # Các loại Agent (Random, Heuristic, XGBoost)
│   ├── train.py                  # Pipeline huấn luyện mô hình
│   └── evaluate.py               # Suite đánh giá hiệu năng
├── scripts/                     # Kịch bản thực thi CLI
│   ├── generate_dataset.py       # Script sinh dataset
│   ├── train_model.py            # Script train model
│   ├── evaluate_agent.py         # Script đánh giá model
│   └── play_gui.py               # Giao diện đồ họa Pygame
├── tests/                       # Bộ kiểm thử tự động (Unit Tests)
├── data/                        # File dữ liệu CSV đã xử lý
├── models/                      # File mô hình đã train (.joblib)
└── results/                     # Kết quả đánh giá và đồ thị
```
