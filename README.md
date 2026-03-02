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

Đánh giá trên tập test độc lập 3.627 mẫu, subject-independent split (80/20 theo danh tính):

| Backbone | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| EfficientNet-V2-B0 | **99.07%** | **99.86%** | **98.34%** | **99.09%** |
| ConvNeXt-Tiny | 94.85% | 97.52% | 94.36% | 95.91% |
| ViT-Base-LoRA | 87.11% | 92.81% | 88.22% | 90.45% |
| Ensemble (Eff + ViT) | ~99.5% | — | — | — |

> [Thêm ảnh biểu đồ so sánh các chỉ số tại đây — Hình 4.1]

Recall theo từng loại tấn công:

| Loại tấn công | ConvNeXt-Tiny | EfficientNet-V2-B0 | ViT-Base-LoRA |
|---|---|---|---|
| Print Attack | 93.98% | 98.83% | 87.96% |
| Replay Attack | 90.50% | 93.95% | 86.83% |
| Mask Attack | 95.37% | 99.74% | 90.61% |
| Deepfake Attack | ~97% | 100% | ~86% |

> [Thêm ảnh Recall theo loại tấn công tại đây — Hình 4.7]

---

## Yêu Cầu Hệ Thống

- Python 3.8+ (tested 3.8/3.9, khuyến nghị 3.11)
- GPU với CUDA (khuyến nghị cho huấn luyện và inference)
- Môi trường thực nghiệm gốc: 3x NVIDIA RTX 3060 Ti (8GB VRAM), AMD EPYC 7K62 48-Core, RAM 378GB, Ubuntu 22.04.3 LTS, CUDA 12.1

---

## Cấu Trúc Dự Án

```
Face-anti-spoof/
├── app.py                           # Demo Streamlit (anh, webcam, video)
├── evaluate_all.py                  # Danh gia toan dien cac mo hinh
├── ensemble_score.py                # Ket hop EfficientNet + ViT
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
│   ├── models/                      # Dinh nghia kien truc mo hinh
│   ├── data/
│   │   ├── data_loader.py
│   │   └── split_dataset.py
│   ├── inference/
│   │   ├── infer_model.py
│   │   └── analyze_test.py
│   └── evaluation/                  # Ham tinh metric, helper plotting
├── data/
│   ├── raw/                         # Du lieu goc tu 6 bo dataset
│   ├── processed/                   # Du lieu sau deep cleaning
│   └── data_process/dataset_split/  # train.csv, test.csv
├── checkpoints/
│   ├── efficientnet/best.pt
│   ├── convnext/best.pt
│   └── vit/best.pt
├── saved_models/                    # Weights dinh dang safetensors
├── scripts/
│   ├── setup_venv.ps1
│   └── data_cleaning/               # Script resize, trich xuat frame, merge nhan
└── results/
    └── final_comparison/            # Bao cao, ROC, confusion matrix
```

---

## Dữ Liệu

Tập dữ liệu tổng hợp từ **6 nguồn quốc tế**, sau khi deep cleaning còn **48.074 mẫu**.

| Nhóm | Dataset | Đặc điểm |
|---|---|---|
| Live | FFHQ | Ảnh chân dung chất lượng cao, đa dạng danh tính |
| Live | VGGFace2 | Ảnh đa dạng pose, biểu cảm, ánh sáng |
| Spoof | CelebA-Spoof | Print, replay, partial/full mask |
| Spoof | FaceForensics++ | Deepfake, Face2Face, FaceSwap, NeuralTextures |
| Spoof | iBeta PAD Level 2 | Mask 3D cao cấp theo chuẩn ISO/IEC 30107-3 |
| Spoof | Silicone Mask Dataset | Mặt nạ silicone custom |

**Không push dữ liệu, model, checkpoint lớn lên GitHub.** Nén thành `data.zip` nếu cần chia sẻ. Khi clone về, giải nén:

```bash
unzip data.zip -d Face-anti-spoof/
```

> [Thêm ảnh phân bố Live/Spoof tại đây — Hình 2.1]

---

## Quy Trình Tiền Xử Lý

### Deep Cleaning

- **Face detection**: MTCNN, ngưỡng tin cậy 0.98, loose crop hệ số **1.4x** để giữ context (mép giấy, viền màn hình, cạnh mặt nạ).
- **Blur filter**: Laplacian variance < 50 bị loại.
- **Deduplication**: pHash, ngưỡng Hamming distance < 5 bit.

### Domain Balancing — Quality Equalization

Để tránh mô hình học "đường tắt" theo chất lượng ảnh, ảnh Live được chủ động hạ chất lượng:

- **JPEG compression** ngẫu nhiên (quality factor 15–45).
- **Gaussian blur** nhẹ (sigma 0.5–1.2).

| Chỉ số | Live gốc (FFHQ/VGGFace2) | Live sau Quality Equalization |
|---|---|---|
| Độ nét (Laplacian) | Rất cao (> 150) | Trung bình (50–90) |
| Nhiễu nén | Gần như không có | Xuất hiện nhiễu khối JPEG |
| Chi tiết vi mô | Studio-level | Mờ nhẹ, giống cảm biến thực tế |

> [Thêm ảnh ví dụ Live/Spoof trước và sau Quality Equalization tại đây — Hình 2.3]

### Identity-Based Split

80% Train / 20% Test theo danh tính — mỗi cá nhân chỉ xuất hiện trong một tập, đảm bảo không rò rỉ thông tin giữa hai tập.

> [Thêm ảnh biểu đồ phân chia train/test tại đây — Hình 2.4]

Chuẩn hoá ảnh trong pipeline:

```
Resize: EfficientNet -> 260x260  |  ConvNeXt / ViT -> 224x224
Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

---

## Kiến Trúc Mô Hình

Ba backbone được kết hợp với **Classification Head thống nhất** (512-dim bottleneck):

```
Backbone Output -> Linear(-> 512) -> Norm -> Activation -> Dropout(0.5-0.6) -> Linear(-> 1) -> Sigmoid
```

Lý do chọn 512 chiều: thử nghiệm với 256/512/1024 cho thấy 512 cho khoảng cách train-val loss nhỏ nhất (~0.08–0.10) và hội tụ mượt nhất.

### ConvNeXt-Tiny

CNN hiện đại hóa theo tinh thần Transformer: kernel 7x7, depthwise convolution, 4 stage, GlobalAvgPool về 768 chiều. 28M tham số.

Regularization đặc thù: Stochastic Depth 0.2 + Dropout tại head.

> [Thêm ảnh kiến trúc ConvNeXt tại đây — Hình 3.1]

### EfficientNet-V2-B0

CNN tối ưu tài nguyên: Fused-MBConv + MBConv + Squeeze-and-Excitation, đầu ra 1280 chiều. 8.4M tham số. Phù hợp triển khai thiết bị biên.

Regularization đặc thù: Dropout 0.6 + Mixup augmentation.

> [Thêm ảnh kiến trúc EfficientNet-V2-B0 tại đây — Hình 3.2]

### ViT-Base + LoRA

Vision Transformer với Parameter-Efficient Fine-Tuning. Ảnh 224x224 chia patch 16x16, chuỗi 197 token qua 12 lớp Transformer Encoder. 86M tham số gốc, chỉ **0.6M trainable** (LoRA rank=16, alpha=32 trên Query và Value).

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

Checkpoint tốt nhất được lưu tự động vào `checkpoints/<model>/best.pt` dựa trên F1-Score validation.

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

### Chiến Lược Huấn Luyện

- **WeightedRandomSampler**: trọng số Live=3.0, Spoof=1.0 để cân bằng tỉ lệ 1:3 trong mỗi batch.
- **Label Smoothing** (0.1–0.2): giảm overconfidence, cải thiện calibration.
- **Progressive Unfreezing**: đóng băng backbone 3–5 epoch đầu (chỉ train head), sau đó mở toàn bộ với learning rate backbone = 1/10 head.
- **Early Stopping** theo F1-Score với patience 10–15 epoch; lưu trọng số tại epoch tốt nhất.

---

## Đánh Giá

```bash
python evaluate_all.py
```

Script load các model từ `checkpoints/*/best.pt`, chạy trên `data/data_process/dataset_split/test.csv`, tính Accuracy, Precision, Recall, F1, AUC, vẽ ROC và Confusion Matrix, lưu kết quả vào `results/final_comparison/`.

> [Thêm ảnh confusion matrix tại đây — Hình 4.6]

> [Thêm ảnh learning curves (Loss + Accuracy) của 3 mô hình tại đây — Hình 4.2, 4.3, 4.4]

> [Thêm ảnh phân bố probability score tại đây — Hình 4.5]

### Ensemble

```bash
python ensemble_score.py
```

Kết hợp xác suất của EfficientNet và ViT theo trọng số mặc định 0.7/0.3.

### Đánh Giá Đa Ngưỡng

Ngoài ngưỡng mặc định 0.5, dự án phân tích thêm:

- **Ngưỡng 0.3** (ưu tiên Recall — chế độ bảo mật cao): phù hợp khi cần phát hiện tối đa tấn công, chấp nhận false alarm cao hơn.
- **Ngưỡng 0.7** (ưu tiên Precision — chế độ thân thiện người dùng): phù hợp khi cần giảm cảnh báo nhầm.

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

Để gọi inference từ code hoặc tích hợp vào hệ thống khác, xem `src/inference/infer_model.py`:

```python
from src.inference.infer_model import load_model, predict_batch

model = load_model("checkpoints/efficientnet/best.pt", backbone="efficientnet")
results = predict_batch(model, image_paths=["face1.jpg", "face2.jpg"])
```

---

## So Sánh Với Nghiên Cứu Gần Đây

| Nghiên cứu | Kiến trúc | Accuracy | F1-Score | Ghi chú |
|---|---|---|---|---|
| Jaswanth et al. (2023) | NLBP-Net | 99.59% | 99.06% | Cần feature engineering thủ công |
| **Luận văn này** | **EfficientNet-V2-B0** | **99.07%** | **99.09%** | End-to-end, precision cao nhất (99.86%) |
| Arti et al. (2023) | Inception-v3 | 98.78% | 99.27% | Dấu hiệu quá khớp |
| Zawar et al. (2023) | MobileNetV2 | 98.00% | 96–99% | Nhẹ, real-time |
| Imam (2024) | AlexNet | 97.98% | 98.93% | Không phân tích chi tiết PAI |

---

## Hạn Chế và Hướng Phát Triển

Hạn chế hiện tại:

- Chỉ sử dụng ảnh RGB, chưa khai thác multi-modal (RGB + Depth + IR).
- Chưa có giao thức cross-database chuyên biệt (ví dụ train CelebA-Spoof, test OULU-NPU).
- ViT-LoRA chưa tích hợp statistical adapter (S-Adapter) hoặc TTDG-FAS.

Hướng phát triển:

- Tích hợp S-Adapter / Test-Time Domain Generalization cho nhánh Transformer.
- Mở rộng đánh giá cross-dataset chuẩn hóa.
- Xây dựng phân loại đa lớp xác định loại tấn công cụ thể (Print / Replay / Mask / Deepfake).
- Bổ sung temporal analysis cho Replay Attack.
- Triển khai multi-modal với depth/IR sensors.

---

## Ghi Chú Kỹ Thuật

- File weights lớn (> 100MB) không upload lên GitHub, lưu local hoặc upload lên Hugging Face / rclone.
- Nếu Streamlit báo lỗi không tìm thấy weights: đặt file vào `checkpoints/<model>/best.pt` hoặc sửa đường dẫn trong `app.py`.
- Nếu dùng GPU, đảm bảo PyTorch cài đúng phiên bản CUDA.
- Một số hàm đọc CSV sẽ lọc các file không tồn tại trong thư mục ảnh — đảm bảo filepath trong `test.csv` khớp với tên file trong `img_dir`.
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
