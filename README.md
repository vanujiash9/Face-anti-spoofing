# 🛡️ Face Anti-Spoofing — So Sánh Kiến Trúc Học Sâu Hiện Đại

**Luận văn tốt nghiệp** · Bùi Thị Thanh Vân (MSSV: 2251320039)  
Ngành Công nghệ Thông tin – Khoa học Dữ liệu · ĐH Giao thông Vận tải TP.HCM  
GVHD: TS. Nguyễn Thị Khánh Tiên  
🔗 [GitHub Repository](https://github.com/vanujiash9/Face-anti-spoof)

---

## 🎯 Bài toán & Đóng góp

Phân loại nhị phân **khuôn mặt thật (Live) vs giả mạo (Spoof)** — bảo vệ hệ thống nhận diện trước 4 kiểu tấn công: **Print · Replay · Mask 3D · Deepfake**.

**Điểm nổi bật:**
- So sánh công bằng **3 backbone hiện đại** trong cùng pipeline & head phân loại
- Tập dữ liệu **~48K mẫu** từ 6 nguồn quốc tế, qua deep cleaning & domain balancing
- Pipeline hoàn chỉnh: dữ liệu → huấn luyện → đánh giá → **demo web Streamlit**

---

## 🗂️ Pipeline & Kiến trúc

### Data Pipeline
![Data Pipeline](results/pipeline/pipeline%20paper-Page-5.drawio.png)

### EfficientNet-V2-B0
![EfficientNet Pipeline](results/pipeline/pipeline%20paper-EfficientNetV2_B0.drawio.png)

### ConvNeXt-Tiny
![ConvNeXt Pipeline](results/pipeline/pipeline%20paper-ConvNext%20Tiny.drawio.png)

### ViT-Base + LoRA
![ViT Pipeline](results/pipeline/pipeline%20paper-ViT-based.drawio.png)

---

## 🏆 Kết quả nổi bật (test set: 3.627 mẫu, subject-independent)

| Backbone | Accuracy | F1-Score | False Positive | False Negative |
|---|---|---|---|---|
| **EfficientNet-V2-B0** | **99.07%** | **99.09%** | **0.21%** | **1.66%** |
| ConvNeXt-Tiny | 94.85% | 95.91% | 3.55% | 5.64% |
| ViT-Base + LoRA | 87.11% | 90.45% | 10.12% | 11.78% |

**EfficientNet-V2-B0** đạt Precision 99.86%, phát hiện **100% Deepfake** và **99.74% Mask 3D** — ngang mức nghiên cứu quốc tế với chỉ **8.4M tham số** (phù hợp triển khai edge).

---

## 🔧 Công nghệ sử dụng

| Hạng mục | Chi tiết |
|---|---|
| **Deep Learning** | PyTorch 2.x, ConvNeXt, EfficientNet-V2, Vision Transformer |
| **Efficient Fine-tuning** | LoRA (rank=16, alpha=32) — giảm trainable params từ 86M → 0.6M |
| **Data Pipeline** | MTCNN · pHash dedup · JPEG compression · Gaussian blur |
| **Training tricks** | WeightedRandomSampler · Mixup · Label smoothing · Progressive unfreezing |
| **Inference & Demo** | Streamlit (ảnh / webcam / video), script inference Python |
| **Hardware** | 3× NVIDIA RTX 3060 Ti · AMD EPYC 48-core · Ubuntu 22.04 · CUDA 12.1 |

---

## 📦 Dataset

| Nguồn | Loại | Vai trò |
|---|---|---|
| FFHQ · VGGFace2 | Live | Đa dạng danh tính, góc, ánh sáng |
| CelebA-Spoof | Print · Replay · Mask | Benchmark chuẩn |
| FaceForensics++ | Deepfake (4 phương pháp) | Face2Face, FaceSwap, ... |
| iBeta PAD Level 2 | Mask 3D cao cấp | Chuẩn ISO/IEC 30107-3 |
| Silicone Mask Dataset | Mask silicon custom | Top-K frame chất lượng cao |

**48.074 mẫu** sau deep cleaning · Split 80/20 **theo danh tính** (subject-independent)

---

## 🚀 Chạy nhanh
```bash
# Cài đặt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train
python src/training/train_efficientnet.py --config config/efficientnet.yaml

# Đánh giá toàn bộ
python evaluate_all.py

# Demo web
streamlit run app.py
```

---

## 📬 Liên hệ

**Bùi Thị Thanh Vân** · thanh.van19062004@gmail.com · [GitHub](https://github.com/vanujiash9/Face-anti-spoof)
