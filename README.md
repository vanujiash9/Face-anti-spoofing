# Face Anti-Spoofing — So Sánh Kiến Trúc Học Sâu Hiện Đại

Luận văn tốt nghiệp · Bùi Thị Thanh Vân · MSSV: 2251320039  
Ngành Công nghệ Thông tin – Chuyên ngành Khoa học Dữ liệu  
Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh  
Giảng viên hướng dẫn: TS. Nguyễn Thị Khánh Tiên  
GitHub: https://github.com/vanujiash9/Face-anti-spoof

---

## Tổng Quan

Dự án nghiên cứu bài toán **phát hiện giả mạo khuôn mặt (Face Anti-Spoofing — FAS)**: phân loại nhị phân xác định khuôn mặt đưa vào hệ thống là **thật (Live)** hay **giả mạo (Spoof)**.

Điểm khác biệt chính:

- **Data-centric approach**: chuẩn hóa và cân bằng chất lượng dữ liệu để tránh shortcut learning, thay vì chỉ tập trung vào kiến trúc mô hình.
- **So sánh công bằng 3 backbone hiện đại** (ConvNeXt-Tiny, EfficientNet-V2-B0, ViT-Base-LoRA) trong cùng một pipeline thống nhất.
- **Đánh giá chi tiết** theo từng loại tấn công: Print, Replay, Mask, Deepfake.
- **Pipeline end-to-end**: từ tiền xử lý dữ liệu, huấn luyện, đánh giá, đến demo web (Streamlit).

---

## Kết Quả

Đánh giá trên tập test độc lập **3.627 mẫu** (1.463 Live và 2.164 Spoof), subject-independent split (80/20 theo danh tính), ngưỡng phân loại 0.5:

| Backbone | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| EfficientNet-V2-B0 | **99.07%** | **99.86%** | **98.34%** | **99.09%** |
| ConvNeXt-Tiny | 94.85% | 97.52% | 94.36% | 95.91% |
| ViT-Base-LoRA | 87.11% | 92.81% | 88.22% | 90.45% |

> [Thêm ảnh biểu đồ so sánh các chỉ số tại đây — Hình 4.1]

### Phân Tích Lỗi (Confusion Matrix)

| Backbone | False Positive (FP) | False Negative (FN) |
|---|---|---|
| ConvNeXt-Tiny | 3.55% (52/1.463 người thật bị nhầm) | 5.64% (122/2.164 tấn công bị bỏ sót) |
| EfficientNet-V2-B0 | **0.21%** (3 trường hợp) | **1.66%** (36 trường hợp) |
| ViT-Base-LoRA | 10.12% (148 trường hợp) | 11.78% (255 trường hợp) |

> [Thêm ảnh confusion matrix tại đây — Hình 4.6]

### Recall Theo Từng Loại Tấn Công

| Loại tấn công | ConvNeXt-Tiny | EfficientNet-V2-B0 | ViT-Base-LoRA |
|---|---|---|---|
| Print Attack | 93.98% | 98.83% | 87.96% |
| Replay Attack | 90.50% | 93.95% | 86.83% |
| Mask Attack | 95.37% | **99.74%** | 90.61% |
| Deepfake Attack | ~97% | **100%** | ~86% |

Tổng hiệu năng theo dải: ConvNeXt-Tiny đồng đều (90.5–97.2%), ViT-Base-LoRA yếu đồng đều (85.7–87.9%).

> [Thêm ảnh biểu đồ Recall theo loại tấn công tại đây — Hình 4.7]

---

## So Sánh Với Nghiên Cứu Gần Đây

| Nghiên cứu | Kiến trúc | Bộ dữ liệu | Accuracy | F1-Score | Ghi chú |
|---|---|---|---|---|---|
| Jaswanth et al. (2023) | NLBP-Net | Đa nguồn | 99.59% | 99.06% | Cần feature engineering thủ công |
| **Luận văn này** | **EfficientNet-V2-B0** | **Đa nguồn (48K)** | **99.07%** | **99.09%** | End-to-end, precision cao nhất 99.86% |
| Arti et al. (2023) | Inception-v3 (mixed6) | MSU-MFSD | 98.78% | 99.27% | Dấu hiệu quá khớp (train 100%) |
| Zawar et al. (2023) | MobileNetV2 | LCC-FASD (18.827 ảnh) | 98.00% | 96–99% | Nhẹ, real-time |
| Imam (2024) | AlexNet | Đa loại tấn công | 97.98% | 98.93% | Không phân tích chi tiết PAI |
| **Luận văn này** | **ConvNeXt-Tiny** | **Đa nguồn (48K)** | **94.85%** | **95.91%** | CNN hiện đại, huấn luyện ổn định |
| **Luận văn này** | **ViT-Base+LoRA** | **Đa nguồn (48K)** | **87.11%** | **90.45%** | Transformer PEFT, cần data lớn hơn |

---

## Yêu Cầu Hệ Thống

- Python 3.8+ (tested 3.8/3.9, khuyến nghị 3.11.9)
- GPU với CUDA (khuyến nghị cho huấn luyện và inference)

Môi trường thực nghiệm gốc:

| Thành phần | Cấu hình |
|---|---|
| GPU | 3x NVIDIA GeForce RTX 3060 Ti (8GB VRAM mỗi card) |
| CPU | AMD EPYC 7K62 48-Core (72 luồng) |
| RAM | 378 GB |
| Storage | 1.3 TB NVMe SSD (~6000/5500 MB/s) |
| OS | Ubuntu 22.04.3 LTS |
| Framework | Python 3.11.9, PyTorch 2.x, CUDA 12.1 |

---

## Cấu Trúc Dự Án

```
Face-anti-spoof/
├── app.py                           # Demo Streamlit (anh, webcam, video)
├── evaluate_all.py                  # Danh gia toan dien cac mo hinh
├── ensemble_score.py                # Ket hop EfficientNet + ViT (0.7/0.3)
├── requirements.txt
├── config/
│   ├── efficientnet.yaml
│   ├── convnext.yaml
│   └── vit.yaml
├── src/
│   ├── training/
│   │   ├── train_efficientnet.py
│   │   ├── train_convnext.py
│   │   └── train_vit.py
│   ├── models/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── split_dataset.py
│   ├── inference/
│   │   ├── infer_model.py
│   │   └── analyze_test.py
│   └── evaluation/
├── data/
│   ├── raw/
│   ├── processed/
│   └── data_process/dataset_split/  # train.csv, test.csv
├── checkpoints/
│   ├── efficientnet/best.pt
│   ├── convnext/best.pt
│   └── vit/best.pt
├── saved_models/
├── scripts/
│   ├── setup_venv.ps1
│   └── data_cleaning/
└── results/
    └── final_comparison/
```

---

## Dữ Liệu

Tập dữ liệu tổng hợp từ **6 nguồn quốc tế**, trước deep cleaning: **51.968 mẫu**; sau deep cleaning còn **48.074 mẫu**.

- Live: 31.991 ảnh (19.991 từ FFHQ + 12.000 từ VGGFace2)
- Spoof: 19.977 mẫu (print, replay, mask 3D, deepfake)
- Tỉ lệ Live/Spoof trong tập huấn luyện: ~24.9% Live / 75.1% Spoof

| Nhóm | Dataset | Đặc điểm |
|---|---|---|
| Live | FFHQ | Ảnh chân dung chất lượng cao, đa dạng danh tính, tuổi, góc nhìn |
| Live | VGGFace2 | Ảnh đa dạng pose, biểu cảm, điều kiện chiếu sáng |
| Spoof | CelebA-Spoof | Print (A4, photo, poster), replay (PC, tablet, phone), partial/full mask |
| Spoof | FaceForensics++ | Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures |
| Spoof | iBeta PAD Level 2 | Mask 3D cao cấp (silicone, latex, wrapped 3D) theo ISO/IEC 30107-3 |
| Spoof | Silicone Mask Dataset | Mặt nạ silicone custom, trích xuất top-K frame theo chất lượng |

**Không push dữ liệu, model, checkpoint lớn lên GitHub.** Nén thành `data.zip` nếu cần chia sẻ. Khi clone về, giải nén:

```bash
unzip data.zip -d Face-anti-spoof/
```

> [Thêm ảnh phân bố Live/Spoof tại đây — Hình 2.1]

---

## Quy Trình Tiền Xử Lý

> [Thêm ảnh pipeline tiền xử lý tại đây — Hình 2.2]

### Deep Cleaning

- **Face detection**: MTCNN, ngưỡng tin cậy 0.98, loose crop hệ số **1.4x** để giữ context (mép giấy, viền màn hình, cạnh mặt nạ).
- **Blur filter**: loại bỏ ảnh có Laplacian variance < 50 (mờ do chuyển động hoặc lỗi lấy nét).
- **Deduplication**: pHash, ngưỡng Hamming distance < 5 bit.

### Domain Balancing — Quality Equalization

Để tránh mô hình học "đường tắt" theo chất lượng ảnh, ảnh Live được chủ động hạ chất lượng:

- **JPEG compression** ngẫu nhiên (quality factor 15–45): tạo nhiễu khối, đưa phân phối Live gần với Spoof.
- **Gaussian blur** nhẹ (sigma 0.5–1.2): giả lập camera tiêu dùng, giữ lại micro-texture cần thiết.

| Chỉ số | Live gốc (FFHQ/VGGFace2) | Live sau Quality Equalization | Mục đích |
|---|---|---|---|
| Độ nét (Laplacian) | Rất cao (> 150) | Trung bình (50–90) | Đồng nhất độ sắc nét với Spoof |
| Nhiễu nén | Gần như không có | Xuất hiện nhiễu khối JPEG | Tránh mô hình dựa vào độ sạch file |
| Chi tiết vi mô | Studio-level | Mờ nhẹ, giống cảm biến thực tế | Ép mô hình học đặc trưng vật liệu da |
| Phân phối miền | Chất lượng rất cao | Gần với camera tiêu dùng | Tăng khả năng tổng quát hóa cross-domain |

> [Thêm ảnh ví dụ Spoof (trái) và Live sau Quality Equalization (phải) tại đây — Hình 2.3]

### Identity-Based Split

80% Train / 20% Test theo danh tính. Mỗi cá nhân chỉ xuất hiện trong một tập — không rò rỉ thông tin giữa hai tập.

> [Thêm ảnh biểu đồ phân chia train/test tại đây — Hình 2.4]

Chuẩn hoá ảnh trong pipeline:

```
Resize: EfficientNet -> 260x260  |  ConvNeXt / ViT -> 224x224
Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

---

## Kiến Trúc Mô Hình

Ba backbone dùng chung **Classification Head thống nhất** (512-dim bottleneck):

```
Backbone Output -> Linear(-> 512) -> Norm -> Activation -> Dropout(0.5-0.6) -> Linear(-> 1) -> Sigmoid
```

Kích thước 512 được chọn sau thử nghiệm 256/512/1024: 512 cho khoảng cách train-val loss nhỏ nhất (~0.08–0.10), hội tụ mượt nhất. 256 gây suy giảm nhẹ hiệu năng (val loss cao hơn ~0.05); 1024 gây overfitting sớm (khoảng cách train-val loss ~0.15).

### ConvNeXt-Tiny

CNN hiện đại hóa theo tinh thần Transformer: kernel 7x7, depthwise convolution, 4 stage, GlobalAvgPool về **768 chiều**. **28M tham số**.

Head: `Linear(768, 512) -> LayerNorm -> ReLU -> Dropout -> Linear(512, 1)`

Regularization đặc thù: Stochastic Depth 0.2 + Dropout tại head. Val Loss thấp hơn Train Loss trong hầu hết các epoch (~0.22 vs ~0.25 tại epoch cuối), hội tụ sau ~17 epoch, plateau ~94.85%.

> [Thêm ảnh kiến trúc ConvNeXt tại đây — Hình 3.1]

### EfficientNet-V2-B0

CNN tối ưu tài nguyên: Fused-MBConv + MBConv + Squeeze-and-Excitation, đầu ra **1280 chiều**. **8.4M tham số**. Phù hợp triển khai thiết bị biên.

Head: `Linear(1280, 512) -> BatchNorm1d -> SiLU -> Dropout(0.6) -> Linear(512, 1)`

Regularization đặc thù: Dropout 0.6 + Mixup augmentation + Label Smoothing 0.2. Khoảng cách train-val accuracy ~1–2%, hội tụ hoàn hảo tại epoch 16–17, plateau 99.07%.

> [Thêm ảnh kiến trúc EfficientNet-V2-B0 tại đây — Hình 3.2]

### ViT-Base + LoRA

Vision Transformer với Parameter-Efficient Fine-Tuning. Ảnh 224x224 chia patch 16x16, chuỗi **197 token** (196 patch + 1 class) qua 12 lớp Transformer Encoder. **86M tham số gốc**, chỉ **0.6M trainable** (LoRA rank=16, alpha=32 trên Query và Value; backbone gốc giữ cố định).

Head: `Linear(768, 512) -> LayerNorm -> GELU -> Dropout -> Linear(512, 1)`

Hội tụ chậm, Val Accuracy tăng từ ~74% lên ~87.11% sau 22 epoch, Val Loss dao động ~±0.03–0.05, Val Loss cuối ~0.27.

> [Thêm ảnh kiến trúc ViT-Base tại đây — Hình 3.3]

---

## Cài Đặt

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Hoặc dùng script tự động:

```powershell
.\scripts\setup_venv.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Huấn Luyện

Chỉnh sửa file cấu hình trong `config/` trước khi chạy (đường dẫn dữ liệu, batch size, learning rate, v.v.).

```bash
# Linux — chạy song song trên 3 GPU
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 src/training/train_convnext.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 src/training/train_efficientnet.py
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 src/training/train_vit.py

# Chỉ định config
python src/training/train_efficientnet.py --config config/efficientnet.yaml
```

Checkpoint tốt nhất được lưu tự động vào `checkpoints/<model>/best.pt` theo F1-Score validation.

### Siêu Tham Số

| Thông số | ConvNeXt-Tiny | EfficientNet-V2-B0 | ViT-Base (LoRA) |
|---|---|---|---|
| Input size | 224x224 | 260x260 | 224x224 |
| Batch size | 64 | 64 | 32 |
| Max epochs | 30 | 40 | 45 |
| Learning rate | 1e-4 | 5e-5 | 5e-5 |
| Weight decay | 0.1 | 0.15 | 0.2 |
| Label smoothing | 0.1 | 0.2 | 0.1 |
| Warm-up epochs | 3 | 5 | 5 |
| Early stopping patience | 10 | 12 | 15 |
| Regularization đặc thù | Stochastic Depth 0.2 | Mixup augmentation | LoRA rank 16 |

### Thời Gian Huấn Luyện và Tài Nguyên

| Backbone | Thời gian/epoch | Tổng thời gian | VRAM | Epoch dừng |
|---|---|---|---|---|
| ConvNeXt-Tiny | 307–308 giây | ~8.300 giây (~2.3 giờ) | ~7.1 GB | 17 |
| EfficientNet-V2-B0 | 124–125 giây (sau warm-up) | ~30–35 phút | ~5.7 GB | 28 (plateau epoch 16–17) |
| ViT-Base-LoRA | 480–485 giây (~8 phút) | ~10.500 giây (~2.9 giờ) với 22 epoch | ~3.0 GB | 22 |

### Chiến Lược Huấn Luyện

- **WeightedRandomSampler**: trọng số Live=3.0, Spoof=1.0 để cân bằng tỉ lệ 1:3 trong mỗi batch, duy trì ~50/50 Live/Spoof.
- **Label Smoothing** (0.1–0.2): thay nhãn cứng {0,1} bằng nhãn mềm (ví dụ {0.05, 0.95}), giảm overconfidence.
- **Progressive Unfreezing**: đóng băng backbone 3–5 epoch đầu (chỉ train Classification Head), sau đó mở toàn bộ với learning rate backbone = 1/10 learning rate head.
- **Early Stopping** theo F1-Score với patience 10–15 epoch; lưu trọng số tại epoch có F1-Score cao nhất.

---

## Đặc Điểm Hội Tụ

| Backbone | Tốc độ hội tụ | Đặc điểm Loss | Plateau Accuracy | Nhận định |
|---|---|---|---|---|
| ConvNeXt-Tiny | Nhanh nhất (17 epoch) | Val < Train (regularization mạnh) | ~94.85% (epoch 12) | Đạt giới hạn sớm |
| EfficientNet-V2-B0 | Trung bình (16–17 epoch) | Song hành hoàn hảo, khoảng cách ~0.10 | **99.07%** (epoch 16) | Hội tụ hoàn hảo |
| ViT-Base-LoRA | Chậm nhất (22+ epoch) | Dao động nhiều, Val Loss cao (~0.27) | ~87% (chưa rõ plateau) | Hội tụ lâu, cải thiện chậm |

> [Thêm ảnh learning curves ConvNeXt tại đây — Hình 4.2]

> [Thêm ảnh learning curves EfficientNet tại đây — Hình 4.3]

> [Thêm ảnh learning curves ViT-Base tại đây — Hình 4.4]

> [Thêm ảnh phân bố probability score tại đây — Hình 4.5]

---

## Đánh Giá

```bash
python evaluate_all.py
```

Load các model từ `checkpoints/*/best.pt`, chạy trên `data/data_process/dataset_split/test.csv`, tính Accuracy, Precision, Recall, F1, AUC, vẽ ROC và Confusion Matrix, lưu vào `results/final_comparison/`.

### Ensemble

```bash
python ensemble_score.py
```

Kết hợp xác suất EfficientNet và ViT theo trọng số mặc định **0.7 EfficientNet + 0.3 ViT**.

### Đánh Giá Đa Ngưỡng

| Ngưỡng | Chế độ | Phù hợp ứng dụng |
|---|---|---|
| 0.3 | Ưu tiên Recall (bảo mật cao) | Phát hiện tối đa tấn công, chấp nhận false alarm cao |
| 0.5 | Cân bằng (mặc định) | Baseline so sánh nghiên cứu |
| 0.7 | Ưu tiên Precision (thân thiện người dùng) | Giảm cảnh báo nhầm, phù hợp eKYC tiêu dùng |

---

## Demo Web

```bash
streamlit run app.py
```

Giao diện hỗ trợ:

- Upload nhiều ảnh (batch), lưu kết quả vào `demo_saved/outputs/`.
- Dùng webcam (Streamlit `camera_input`).
- Upload video, xử lý từng frame, lưu video kết quả.

Hệ thống tự động phát hiện khuôn mặt, phân loại live/spoof, vẽ bounding box và hiển thị xác suất dự đoán.

---

## Inference Từ Script

Xem `src/inference/infer_model.py`:

```python
from src.inference.infer_model import load_model, predict_batch

model = load_model("checkpoints/efficientnet/best.pt", backbone="efficientnet")
results = predict_batch(model, image_paths=["face1.jpg", "face2.jpg"])
```

---

## Hạn Chế và Hướng Phát Triển

Hạn chế hiện tại:

- Chỉ sử dụng ảnh RGB, chưa khai thác multi-modal (RGB + Depth + IR).
- Chưa có giao thức cross-database chuyên biệt (ví dụ train CelebA-Spoof, test OULU-NPU/SiW).
- ViT-LoRA chưa tích hợp statistical adapter (S-Adapter) hoặc TTDG-FAS; LoRA rank 16 chưa đủ với 48K mẫu.
- Replay Attack là điểm yếu chung của cả 3 mô hình (recall thấp nhất trong 4 loại tấn công).

Hướng phát triển:

- Tích hợp S-Adapter / Test-Time Domain Generalization cho nhánh Transformer.
- Mở rộng đánh giá cross-dataset chuẩn hóa.
- Xây dựng phân loại đa lớp xác định loại tấn công cụ thể (Print / Replay / Mask / Deepfake).
- Bổ sung temporal analysis hoặc multi-modal fusion với depth/IR sensors cho Replay Attack.
- Thu thập thêm dữ liệu: deepfake chất lượng cao, mặt nạ silicon y học, phát lại trên màn hình OLED.

---

## Ghi Chú Kỹ Thuật

- File weights lớn (> 100MB) không upload lên GitHub, lưu local hoặc upload lên Hugging Face / rclone.
- Nếu Streamlit báo lỗi không tìm thấy weights: đặt file vào `checkpoints/<model>/best.pt` hoặc sửa đường dẫn trong `app.py`.
- Nếu dùng GPU, đảm bảo PyTorch cài đúng phiên bản CUDA.
- Một số hàm đọc CSV sẽ lọc các file không tồn tại trong thư mục ảnh — đảm bảo filepath trong `test.csv` khớp với `img_dir`.
- Đảm bảo đường dẫn file trong `config/*.yaml` đúng với hệ thống của bạn.

---

## Đóng Góp

Mọi contribution (issue, PR) đều hoan nghênh. Trước khi đóng góp:

- Mô tả rõ mục tiêu (bugfix / feature mới).
- Nếu thêm model mới, cập nhật `evaluate_all.py` và `ensemble_score.py` để include model.

---

## Liên Hệ

Bùi Thị Thanh Vân  
Email: thanh.van19062004@gmail.com  
GitHub: https://github.com/vanujiash9/Face-anti-spoof
