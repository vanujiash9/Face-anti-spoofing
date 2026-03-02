# Face Anti-Spoofing — So Sánh Kiến Trúc Học Sâu Hiện Đại

Luận văn tốt nghiệp – Bùi Thị Thanh Vân (MSSV: 2251320039)  
Ngành Công nghệ Thông tin – Chuyên ngành Khoa học Dữ liệu  
Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh  
Giảng viên hướng dẫn: TS. Nguyễn Thị Khánh Tiên  

GitHub: https://github.com/vanujiash9/Face-anti-spoof  

***

## 1. Giới thiệu

Dự án nghiên cứu bài toán **Face Anti-Spoofing (FAS)**, phân loại nhị phân xác định một khuôn mặt đưa vào hệ thống là **thật (Live)** hay **giả mạo (Spoof)**.  
Bốn nhóm tấn công chính được xem xét: **Print Attack, Replay Attack, Mask Attack, Digital/Deepfake Attack**.

Điểm khác biệt của dự án:

- **Data-centric**: ưu tiên chuẩn hóa, làm sạch và cân bằng miền dữ liệu để tránh shortcut learning.  
- **So sánh công bằng 3 backbone hiện đại** (ConvNeXt-Tiny, EfficientNet-V2-B0, ViT-Base-LoRA) trong cùng một pipeline, cùng head phân loại.  
- **Đánh giá chi tiết theo từng loại tấn công**: Print, Replay, Mask 3D, Deepfake.  
- **Pipeline end-to-end**: từ chuẩn bị dữ liệu, huấn luyện, đánh giá, đến demo web (Streamlit) và script inference.

***

## 2. Kết quả chính

Đánh giá trên tập test độc lập **3.627 mẫu** (1.463 Live, 2.164 Spoof), subject-independent split 80/20 theo danh tính, ngưỡng 0.5.

### 2.1. Hiệu năng tổng thể

| Backbone            | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| EfficientNet-V2-B0  | **99.07%** | **99.86%** | **98.34%** | **99.09%** |
| ConvNeXt-Tiny       | 94.85%   | 97.52%    | 94.36% | 95.91%   |
| ViT-Base-LoRA       | 87.11%   | 92.81%    | 88.22% | 90.45%   |

### 2.2. Phân tích lỗi

| Backbone           | False Positive (FP)                | False Negative (FN)                    |
|--------------------|------------------------------------|----------------------------------------|
| ConvNeXt-Tiny      | 3.55% (52 người thật bị nhầm)     | 5.64% (122 tấn công bị bỏ sót)        |
| EfficientNet-V2-B0 | **0.21%** (3 trường hợp)          | **1.66%** (36 trường hợp)             |
| ViT-Base-LoRA      | 10.12% (148 trường hợp)           | 11.78% (255 trường hợp)               |

### 2.3. Recall theo loại tấn công

| Loại tấn công   | ConvNeXt-Tiny | EfficientNet-V2-B0 | ViT-Base-LoRA |
|-----------------|---------------|---------------------|---------------|
| Print Attack    | 93.98%        | 98.83%              | 87.96%        |
| Replay Attack   | 90.50%        | 93.95%              | 86.83%        |
| Mask Attack     | 95.37%        | **99.74%**          | 90.61%        |
| Deepfake Attack | ~97%          | **100%**            | ~86%          |

EfficientNet-V2-B0 cho hiệu năng cao và ổn định nhất, ConvNeXt-Tiny cân bằng tốt, ViT-Base-LoRA cần thêm dữ liệu/adapter để phát huy.

***

## 3. Dataset và tiền xử lý

### 3.1. Tổng quan dữ liệu

Tập dữ liệu tổng hợp từ **6 nguồn quốc tế**.  
- Trước deep cleaning: **51.968 mẫu**  
- Sau deep cleaning: **48.074 mẫu**

Phân bố:

- **Live**: 31.991 ảnh  
  - FFHQ (chân dung chất lượng cao, đa dạng danh tính, tuổi, góc nhìn)  
  - VGGFace2 (đa dạng pose, biểu cảm, ánh sáng)  
- **Spoof**: 19.977 mẫu  
  - CelebA-Spoof (Print, Replay, Mask)  
  - FaceForensics++ (Deepfake: Face2Face, FaceSwap, FaceShifter, NeuralTextures)  
  - iBeta PAD Level 2 (Mask 3D cao cấp, chuẩn ISO/IEC 30107-3)  
  - Silicone Mask Dataset (mặt nạ silicone custom, chọn top-K frame chất lượng cao)  

Tỉ lệ Live/Spoof trong tập train: ~24.9% Live / 75.1% Spoof (cân bằng lại bằng sampler trong quá trình huấn luyện).

> Lưu ý: Dữ liệu và model không được đẩy lên GitHub. Khi cần, nén thành `data.zip` và giải nén vào thư mục gốc dự án.

```bash
unzip data.zip -d Face-anti-spoof/
```

### 3.2. Deep cleaning & split

- **Face detection**: MTCNN, ngưỡng tin cậy 0.98, loose crop ~1.4x để giữ context (mép giấy, viền màn hình, cạnh mặt nạ).  
- **Blur filter**: loại ảnh mờ với Laplacian variance < 50.  
- **Deduplication**: pHash, loại các ảnh có Hamming distance < 5 bit.  
- **Identity-based split**: 80% Train / 20% Test theo danh tính, mỗi người chỉ xuất hiện ở một tập.

### 3.3. Domain balancing (Quality Equalization)

Để tránh mô hình học shortcut theo chất lượng ảnh:

- Áp dụng **JPEG compression** ngẫu nhiên (quality 15–45) trên ảnh Live.  
- Áp dụng **Gaussian blur** nhẹ (sigma 0.5–1.2) mô phỏng camera tiêu dùng.  

Kết quả: phân phối độ nét và nhiễu nén của Live gần hơn với Spoof, buộc mô hình tập trung vào vật liệu da và dấu hiệu spoof thay vì “độ sạch” file.

### 3.4. Chuẩn hóa và kích thước input

- Resize:  
  - EfficientNet-V2-B0 → 260×260  
  - ConvNeXt-Tiny / ViT-Base → 224×224  
- Normalize: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

***

## 4. Kiến trúc mô hình

Ba backbone chia sẻ chung một **classification head** 512 chiều:

```text
Backbone Output
    -> Linear(·, 512)
    -> Norm (BatchNorm/LayerNorm)
    -> Activation
    -> Dropout(0.5–0.6)
    -> Linear(512, 1)
    -> Sigmoid
```

Kích thước 512 được chọn sau khi so sánh 256/512/1024 (512 cho train–val loss gap nhỏ và hội tụ ổn định nhất).

### 4.1. ConvNeXt-Tiny

- CNN hiện đại lấy cảm hứng từ Transformer (kernel 7×7, depthwise conv, 4 stage).  
- Output feature: **768 chiều**, ~**28M** tham số.  
- Head: `Linear(768, 512) -> LayerNorm -> ReLU -> Dropout -> Linear(512, 1)`.  
- Regularization: **Stochastic Depth 0.2** + Dropout tại head.

### 4.2. EfficientNet-V2-B0

- CNN tối ưu tài nguyên: Fused-MBConv + MBConv + Squeeze-and-Excitation.  
- Output feature: **1280 chiều**, ~**8.4M** tham số (phù hợp deployment edge).  
- Head: `Linear(1280, 512) -> BatchNorm1d -> SiLU -> Dropout(0.6) -> Linear(512, 1)`.  
- Regularization: Dropout 0.6, **Mixup augmentation**, **Label smoothing 0.2**.

### 4.3. ViT-Base + LoRA

- Vision Transformer: ảnh 224×224 chia thành patch 16×16, chuỗi **197 token** (196 patch + 1 class), 12 encoder layers.  
- Tham số gốc: ~**86M**, trainable chỉ ~**0.6M** nhờ **LoRA** (rank=16, alpha=32 trên Query & Value).  
- Head: `Linear(768, 512) -> LayerNorm -> GELU -> Dropout -> Linear(512, 1)`.  

***

## 5. Cấu trúc dự án

```text
Face-anti-spoof/
├── app.py                        # Demo Streamlit (ảnh, webcam, video)
├── evaluate_all.py               # Đánh giá toàn bộ mô hình trên test
├── ensemble_score.py             # Ensemble EfficientNet + ViT (trọng số 0.7 / 0.3)
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
│   └── data_process/dataset_split/   # train.csv, test.csv
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

***

## 6. Cài đặt & chạy

### 6.1. Yêu cầu hệ thống

- Python **3.8+** (khuyến nghị 3.11.9).  
- GPU có CUDA (khuyến nghị cho cả training và inference).  

Môi trường thực nghiệm gốc:

- 3× NVIDIA RTX 3060 Ti (8GB VRAM mỗi card)  
- AMD EPYC 7K62 48-Core (72 luồng), 378 GB RAM  
- 1.3 TB NVMe SSD  
- Ubuntu 22.04.3 LTS, PyTorch 2.x, CUDA 12.1  

### 6.2. Cài đặt môi trường

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# Hoặc dùng script
.\scripts\setup_venv.ps1
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

***

## 7. Huấn luyện mô hình

Chỉnh sửa đường dẫn và siêu tham số trong `config/*.yaml` trước khi train.

Ví dụ (Linux, chạy song song 3 GPU):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 src/training/train_convnext.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 src/training/train_efficientnet.py
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 src/training/train_vit.py

# Hoặc chỉ định config cụ thể
python src/training/train_efficientnet.py --config config/efficientnet.yaml
```

Checkpoint tốt nhất (theo F1-Score validation) được lưu tại:
`checkpoints/<model>/best.pt`.

### 7.1. Siêu tham số chính

| Thông số              | ConvNeXt-Tiny | EfficientNet-V2-B0 | ViT-Base (LoRA) |
|-----------------------|---------------|---------------------|-----------------|
| Input size            | 224×224       | 260×260             | 224×224         |
| Batch size            | 64            | 64                  | 32              |
| Max epochs            | 30            | 40                  | 45              |
| Learning rate         | 1e-4          | 5e-5                | 5e-5            |
| Weight decay          | 0.1           | 0.15                | 0.2             |
| Label smoothing       | 0.1           | 0.2                 | 0.1             |
| Warm-up epochs        | 3             | 5                   | 5               |
| Early stopping        | 10            | 12                  | 15              |
| Regularization đặc thù| StochasticDepth 0.2 | Mixup           | LoRA rank 16    |

### 7.2. Chiến lược huấn luyện

- **WeightedRandomSampler**: trọng số Live = 3.0, Spoof = 1.0 để giữ tỷ lệ gần 50/50 trong batch.  
- **Label smoothing** 0.1–0.2 giảm overconfidence.  
- **Progressive unfreezing**:  
  - 3–5 epoch đầu: chỉ train classification head (backbone frozen).  
  - Sau đó mở toàn bộ backbone với LR backbone = 1/10 LR head.  
- **Early stopping theo F1-Score** (patience 10–15), lưu best checkpoint.

### 7.3. Thời gian huấn luyện (tham khảo)

| Backbone         | Thời gian/epoch | Tổng thời gian       | VRAM      | Epoch dừng |
|------------------|-----------------|----------------------|-----------|-----------|
| ConvNeXt-Tiny    | ~307–308 giây   | ~8.300 giây (~2.3h)  | ~7.1 GB   | 17        |
| EfficientNet-V2-B0 | ~124–125 giây | ~30–35 phút          | ~5.7 GB   | 28        |
| ViT-Base-LoRA    | ~480–485 giây   | ~10.500 giây (~2.9h) | ~3.0 GB   | 22        |

***

## 8. Đánh giá & ensemble

### 8.1. Đánh giá toàn diện

```bash
python evaluate_all.py
```

Script sẽ:

- Load các model từ `checkpoints/*/best.pt`.  
- Chạy trên `data/data_process/dataset_split/test.csv`.  
- Tính Accuracy, Precision, Recall, F1, AUC.  
- Vẽ ROC, Confusion Matrix.  
- Lưu kết quả vào `results/final_comparison/`.

### 8.2. Ensemble EfficientNet + ViT

```bash
python ensemble_score.py
```

Mặc định kết hợp xác suất theo trọng số:

- 0.7 × EfficientNet-V2-B0  
- 0.3 × ViT-Base-LoRA  

### 8.3. Đánh giá đa ngưỡng

| Ngưỡng | Chế độ                   | Ứng dụng gợi ý               |
|--------|--------------------------|------------------------------|
| 0.3    | Ưu tiên Recall           | Bảo mật cao, chấp nhận FP    |
| 0.5    | Cân bằng (mặc định)     | Baseline nghiên cứu          |
| 0.7    | Ưu tiên Precision        | eKYC, trải nghiệm người dùng |

***

## 9. Inference & demo

### 9.1. Script inference

Ví dụ sử dụng EfficientNet-V2-B0:

```python
from src.inference.infer_model import load_model, predict_batch

model = load_model("checkpoints/efficientnet/best.pt", backbone="efficientnet")
results = predict_batch(model, image_paths=["face1.jpg", "face2.jpg"])

for path, score in results:
    print(path, score)   # score ~ P(Live)
```

### 9.2. Demo web (Streamlit)

```bash
streamlit run app.py
```

Tính năng:

- Upload nhiều ảnh, lưu kết quả vào `demo_saved/outputs/`.  
- Chụp trực tiếp từ webcam (`camera_input`).  
- Upload video, xử lý frame-by-frame, lưu video kết quả.  

Hệ thống tự động:

- Phát hiện khuôn mặt.  
- Phân loại Live/Spoof.  
- Vẽ bounding box và hiển thị xác suất dự đoán.

***

## 10. So sánh với nghiên cứu gần đây (tóm tắt)

| Nghiên cứu       | Kiến trúc             | Dataset                  | Accuracy | F1-Score | Ghi chú                          |
|------------------|-----------------------|--------------------------|----------|----------|----------------------------------|
| Jaswanth et al.  | NLBP-Net              | Đa nguồn                 | 99.59%   | 99.06%   | Cần feature engineering thủ công |
| Arti et al.      | Inception-v3 (mixed6) | MSU-MFSD                 | 98.78%   | 99.27%   | Dấu hiệu overfit (train 100%)    |
| Zawar et al.     | MobileNetV2           | LCC-FASD                 | 98.00%   | 96–99%   | Nhẹ, real-time                   |
| Imam             | AlexNet               | Đa loại tấn công         | 97.98%   | 98.93%   | Không phân tích chi tiết PAI     |
| **Luận văn này** | **EffNet-V2-B0**      | **Đa nguồn (~48K)**      | **99.07%** | **99.09%** | End-to-end, precision 99.86%    |
| **Luận văn này** | **ConvNeXt-Tiny**     | **Đa nguồn (~48K)**      | 94.85%   | 95.91%   | CNN hiện đại, huấn luyện ổn định |
| **Luận văn này** | **ViT-Base+LoRA**     | **Đa nguồn (~48K)**      | 87.11%   | 90.45%   | PEFT, cần data lớn hơn           |

***

## 11. Hạn chế & hướng phát triển

**Hạn chế hiện tại:**

- Chỉ dùng ảnh RGB, chưa khai thác multi-modal (RGB + Depth + IR).  
- Chưa đánh giá cross-database chuyên biệt (ví dụ train CelebA-Spoof, test OULU-NPU/SiW).  
- Nhánh ViT-LoRA chưa tích hợp S-Adapter, TTDG-FAS; LoRA rank 16 có thể còn nhỏ so với 48K mẫu.  
- Replay Attack là điểm yếu chung (recall thấp nhất trong 4 loại tấn công).

**Hướng phát triển:**

- Tích hợp **S-Adapter / Test-Time Domain Generalization** cho Transformer.  
- Thiết kế protocol **cross-dataset** chuẩn.  
- Mở rộng sang **phân loại đa lớp** (Print / Replay / Mask / Deepfake).  
- Bổ sung **temporal analysis** hoặc **multi-modal fusion** (depth/IR) để xử lý Replay tốt hơn.  
- Thu thập thêm dữ liệu: deepfake chất lượng cao, mặt nạ silicon y học, thiết bị màn hình hiện đại (OLED).

***

## 12. Ghi chú kỹ thuật

- File weights lớn (> 100MB) không được upload trực tiếp lên GitHub; nên lưu local hoặc trên Hugging Face/rclone và tham chiếu đường dẫn.  
- Nếu Streamlit báo không tìm thấy weights:  
  - Đặt file vào `checkpoints/<model>/best.pt`, hoặc  
  - Sửa đường dẫn tương ứng trong `app.py`.  
- Đảm bảo phiên bản PyTorch tương thích với CUDA trên máy.  
- Một số hàm đọc CSV sẽ bỏ qua sample nếu path không tồn tại – đảm bảo `test.csv` khớp với thư mục ảnh.  
- Kiểm tra kỹ đường dẫn dữ liệu trong `config/*.yaml` khi deploy trên máy mới.

***

## 13. Đóng góp & liên hệ

**Đóng góp**

Mọi đóng góp (issue, PR) đều được chào đón. Khi mở PR:

- Mô tả rõ mục tiêu (bugfix / feature mới / mô hình mới).  
- Nếu thêm model mới, cập nhật `evaluate_all.py` và `ensemble_score.py` để include model trong pipeline so sánh.

**Liên hệ**

- Tác giả: **Bùi Thị Thanh Vân**  
- Email: thanh.van19062004@gmail.com  
- GitHub: https://github.com/vanujiash9/Face-anti-spoof
