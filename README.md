
# Face Liveness Detection (PAD)

**Tác giả:** Bùi Thị Thanh Vân

## Mục tiêu
Hệ thống phát hiện tấn công giả mạo khuôn mặt (Face Liveness Detection) bằng deep learning, phân biệt live/spoof từ ảnh hoặc video.

## Tính năng chính
- Huấn luyện và đánh giá các backbone: EfficientNet, ConvNeXt, ViT
- Pipeline end-to-end: tiền xử lý, huấn luyện, inference, ensemble, báo cáo
- Demo trực quan với Streamlit (`app.py`)
- Hỗ trợ upload/checkpoint, lưu kết quả, vẽ biểu đồ

## Cài đặt nhanh
```bash
# Tạo môi trường ảo (nên dùng)
python -m venv .venv
source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt

# Chạy demo giao diện
streamlit run app.py
```

## Cấu trúc thư mục
- `app.py` : Giao diện demo
- `ensemble_score.py`, `evaluate_all.py` : Script ensemble, đánh giá
- `requirements.txt` : Thư viện cần thiết
- `config/` : File YAML cấu hình backbone
- `checkpoints/` : Lưu weights từng model (best.pt, last.pt)
- `saved_models/` : Model lưu sẵn (safetensors, config)
- `src/` : Mã nguồn chính
  - `training/` : Script huấn luyện
  - `inference/` : Hàm inference, phân tích
  - `models/` : Định nghĩa model
  - `data/` : Loader, tiền xử lý
  - `evaluation/` : Hàm tính metric, vẽ biểu đồ
- `scripts/` : Script tiện ích, tiền xử lý dữ liệu

## Hướng dẫn sử dụng
### Huấn luyện
```bash
python src/training/train_efficientnet.py --config config/efficientnet.yaml
```
- Thay đổi backbone hoặc config YAML tùy ý.

### Đánh giá & Ensemble
```bash
python evaluate_all.py
python ensemble_score.py
```

### Inference/Demo
- Chạy `streamlit run app.py` để demo trực quan (ảnh, webcam, video)
- Hoặc dùng `src/inference/infer_model.py` để gọi từ script

## Lưu ý
- File weights lớn (>100MB) không upload lên GitHub, chỉ lưu local hoặc upload lên Hugging Face/rclone.
- Đảm bảo đường dẫn file trong config đúng với hệ thống của bạn.

## Đóng góp & Liên hệ
- Mọi đóng góp, câu hỏi, hoặc yêu cầu mở rộng vui lòng liên hệ qua GitHub hoặc email.

---
README này đã được rút gọn, tập trung hướng dẫn sử dụng và cấu trúc dự án. Nếu cần bổ sung chi tiết về từng script/config, hãy phản hồi thêm!

2. **Huấn luyện mô hình**

- Chọn backbone (EfficientNet, ConvNeXt, ViT).
- Cấu hình tham số huấn luyện qua file YAML (learning rate, batch size, epochs, v.v.).
- Chạy script huấn luyện tương ứng trong `src/training/`.
- Lưu checkpoint tốt nhất vào `checkpoints/<model>/best.pt`.

3. **Đánh giá mô hình**

- Sử dụng script `evaluate_all.py` để đánh giá các mô hình trên tập test.
- Tính toán các chỉ số: Accuracy, Precision, Recall, F1, AUC, vẽ ROC/Confusion Matrix.
- Lưu báo cáo và hình ảnh vào `src/results/final_comparison/`.

4. **Ensemble (Kết hợp mô hình)**

- Chạy script `ensemble_score.py` để kết hợp kết quả từ EfficientNet và ViT (trọng số mặc định 0.7/0.3).
- So sánh kết quả ensemble với từng mô hình đơn lẻ.

5. **Inference/Demo thực tế**

- Chạy `app.py` với Streamlit để demo trực quan:
  - Upload ảnh/video hoặc dùng webcam.
  - Hệ thống tự động phát hiện khuôn mặt, phân loại live/spoof, vẽ khung và lưu kết quả.
- Có thể gọi inference từ script hoặc tích hợp vào hệ thống lớn hơn.

6. **Báo cáo & tổng kết**

- Tổng hợp kết quả, phân tích ưu nhược điểm từng mô hình.
- Đề xuất hướng cải tiến (bổ sung data, thử backbone mới, tuning hyperparameter, v.v.).

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt và chạy nhanh (Windows PowerShell)](#cài-đặt-và-chạy-nhanh-windows-powershell)
- [Dữ liệu và tiền xử lý](#dữ-liệu-và-tiền-xử-lý)
- [Huấn luyện](#huấn-luyện)
- [Inference / Demo](#inference--demo)
- [Đánh giá & So sánh mô hình](#đánh-giá--so-sánh-mô-hình)
- [Checkpoints & Saved Models](#checkpoints--saved-models)
- [Scripts tiện ích](#scripts-tiện-ích)
- [Ghi chú / Troubleshooting](#ghi-chú--troubleshooting)
- [Đóng góp](#đóng-góp)
- [Liên hệ](#liên-hệ)

---

## Tổng quan

Dự án triển khai các mô hình deep learning (EfficientNet, ConvNeXt, ViT) cho bài toán phát hiện "face liveness" — phân biệt ảnh/video người thật (live) và tấn công giả mạo (spoof). Bao gồm:

- Mã huấn luyện, kiểm thử và đánh giá.
- Môi trường demo Streamlit để chạy inference trên ảnh, webcam, video (`app.py`).
- Tập lệnh đánh giá, ensemble và so sánh kết quả (`evaluate_all.py`, `ensemble_score.py`).

Mục tiêu: cung cấp pipeline end-to-end từ tiền xử lý dữ liệu -> huấn luyện -> inference -> báo cáo kết quả.

## Yêu cầu hệ thống

- Python 3.8+ (tested with 3.8/3.9)
- GPU được khuyến nghị (CUDA) cho huấn luyện / inference nhanh
- Các thư viện chính: `torch`, `torchvision`, `timm`, `safetensors`, `pandas`, `pyyaml`, `opencv-python`, `Pillow`.

Các dependency chính đã lưu trong `requirements.txt`.

## Cấu trúc dự án (chỉ nêu các file/folder quan trọng)

- `app.py` : Demo Streamlit (upload ảnh, webcam, video).
- `ensemble_score.py` : Script chạy ensemble (EfficientNet + ViT) để lấy kết quả cuối cùng.
- `evaluate_all.py` : Script đánh giá các mô hình, vẽ ROC/CM và xuất báo cáo.
- `requirements.txt` : Danh sách các package cần cài.
- `config/` : Các file cấu hình YAML mặc định cho từng backbone (`efficientnet.yaml`, `vit.yaml`, `convnet.yaml`).
- `checkpoints/` : Nơi lưu `best.pt` / `last.pt` cho từng backbone.
- `saved_models/` : Một số model đã lưu sẵn (định dạng safetensors / config).
- `scripts/` : Script tiện ích, ví dụ `setup_venv.ps1`, và các script tiền xử lý dữ liệu.
- `src/` : Mã nguồn chính:
  - `src/training/` : script huấn luyện (ví dụ `train_vit.py`, `train_convnext.py`, `train_efficientnet.py`).
  - `src/inference/` : file helper để inference/analysis (`infer_model.py`, `analyze_test.py`).
  - `src/models/` : cấu trúc mô hình, scripts dựng model.
  - `src/data/` : loader/tiền xử lý dữ liệu.
  - `src/evaluation/` : hàm tính metric, helper plotting.

## Giải thích chi tiết các file chính

### 1. app.py

Giao diện demo bằng Streamlit. Cho phép upload ảnh, video, dùng webcam để kiểm tra liveness. Tự động phát hiện khuôn mặt, phân loại live/spoof, vẽ khung và lưu kết quả. Phù hợp trình diễn trực quan hoặc kiểm thử nhanh.

### 2. evaluate_all.py

Script đánh giá toàn diện các mô hình (EfficientNet, ConvNeXt, ViT) trên tập test. Tính toán các chỉ số (Accuracy, Precision, Recall, F1, AUC), vẽ biểu đồ ROC, Confusion Matrix, xuất báo cáo chi tiết vào thư mục `src/results/final_comparison/`.

### 3. ensemble_score.py

Script thực hiện kết hợp (ensemble) hai mô hình EfficientNet và ViT bằng cách lấy trung bình trọng số xác suất dự đoán. Giúp tăng độ chính xác so với dùng một mô hình đơn lẻ.

### 4. src/training/

Chứa các script huấn luyện cho từng backbone:

- `train_efficientnet.py`: Huấn luyện mô hình EfficientNet.
- `train_convnext.py`: Huấn luyện ConvNeXt.
- `train_vit.py`: Huấn luyện ViT.
  Các script này nhận tham số cấu hình từ file YAML, lưu checkpoint tốt nhất vào `checkpoints/`.

### 5. src/inference/

Chứa các hàm, script hỗ trợ inference (dự đoán) trên tập dữ liệu hoặc từng ảnh/video. Ví dụ:

- `infer_model.py`: Hàm load model, chạy dự đoán batch.
- `analyze_test.py`: Phân tích kết quả test, xuất thống kê.

### 6. src/data/

Chứa các module xử lý dữ liệu:

- `data_loader.py`: Định nghĩa dataset, hàm load ảnh và nhãn từ CSV.
- `split_dataset.py`: Hỗ trợ chia train/test, lưu file CSV.

### 7. config/

Chứa các file YAML cấu hình cho từng backbone (EfficientNet, ConvNeXt, ViT). Các trường chính gồm: đường dẫn dữ liệu, batch size, learning rate, số epoch, v.v. Người dùng chỉnh sửa file này để thay đổi tham số huấn luyện.

### 8. checkpoints/

Lưu các file trọng số (weights) tốt nhất và cuối cùng cho từng mô hình (`best.pt`, `last.pt`). Khi inference hoặc đánh giá, script sẽ tự động load file này.

### 9. saved_models/

Chứa các model đã lưu sẵn ở định dạng safetensors và file config. Dùng để chuyển đổi hoặc tải nhanh mô hình mà không cần huấn luyện lại.

### 10. scripts/

Chứa các script tiện ích:

- `setup_venv.ps1`: Tạo môi trường ảo và cài dependencies.
- `data_cleaning/`: Các script tiền xử lý dữ liệu như resize ảnh, trích xuất frame, thống kê kích thước, merge file nhãn, v.v.

---

## Cài đặt và chạy nhanh (Windows PowerShell)

1. Tạo virtual environment và kích hoạt (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Cài dependencies:

```powershell
pip install -r requirements.txt
```

3. Chạy demo Streamlit (mở trình duyệt):

```powershell
streamlit run app.py
```

4. Chạy đánh giá mô hình (sẽ lưu kết quả vào `src/results/final_comparison`):

```powershell
python evaluate_all.py
```

5. Chạy ensemble (Eff + ViT):

```powershell
python ensemble_score.py
```

Lưu ý: Trước khi chạy các script đánh giá/ensemble, cần đảm bảo file weight (`checkpoints/*/best.pt`) có tồn tại.

## Dữ liệu và tiền xử lý

- Thư mục dữ liệu chính nằm ở `data/` (các file tiền xử lý nằm trong `scripts/data_cleaning/`).
- Kết quả tách train/test được lưu trong `data/data_process/dataset_split/` (ví dụ `test.csv`).
- `src/data/data_loader.py` và `src/data/split_dataset.py` chứa logic tạo dataset và loader.

Chuẩn hoá ảnh được sử dụng trong pipeline:

- Resize theo kích thước model (Eff: 260, ViT/ConvNeXt: 224)
- Normalize bằng mean/std ImageNet `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

## Huấn luyện

- Các script huấn luyện nằm trong `src/training/`:
  - `train_efficientnet.py`, `train_convnext.py`, `train_vit.py`
- Các cấu hình huấn luyện (learning rate, batch size, đường dẫn dữ liệu) được lưu trong `config/*.yaml` — chỉnh sửa trước khi chạy.

Một ví dụ chạy (PowerShell):

```powershell
python src/training/train_efficientnet.py --config config/efficientnet.yaml
```

Tuỳ theo script, có thể có các tham số CLI khác (kiểm tra header hoặc `argparse` trong file).

## Inference / Demo

- `app.py` là front-end bằng Streamlit cho phép:

  - Upload nhiều ảnh (batch) và lưu kết quả vào `demo_saved/outputs`.
  - Sử dụng webcam (Streamlit `camera_input`).
  - Upload video, xử lý frame và lưu video kết quả.

- Nếu muốn gọi inference từ CLI/Script: xem `src/inference/infer_model.py` để biết cách load model và chạy batch inference.

## Đánh giá & So sánh mô hình

- `evaluate_all.py`:

  - Load các model từ `checkpoints/*/best.pt` (Eff, ConvNeXt, ViT).
  - Chạy trên `test.csv` (đường dẫn lấy từ `config/*`), xuất metrics: Accuracy, Precision, Recall, F1, AUC.
  - Lưu biểu đồ ROC & Confusion Matrix vào `src/results/final_comparison`.

- `ensemble_score.py`:
  - Chạy inference riêng cho EfficientNet và ViT ở kích thước tương ứng, sau đó kết hợp probability theo trọng số (mặc định 0.7 Eff + 0.3 ViT).

## Checkpoints & Saved Models

- Checkpoints mặc định: `checkpoints/efficientnet/best.pt`, `checkpoints/vit/best.pt`, `checkpoints/convnext/best.pt`.
- Thư mục `saved_models/` chứa model ở dạng `safetensors` và config tương ứng (dùng khi muốn chuyển sang tải sẵn weights khác).

Nếu bạn không có checkpoints, cần huấn luyện lại hoặc thay thế bằng các model trong `saved_models/` (nếu có hỗ trợ loader tương ứng).

## Scripts tiện ích

- `scripts/setup_venv.ps1` : script PowerShell tạo venv và cài dependencies.
- `scripts/data_cleaning/` : nhiều script hỗ trợ trích xuất ảnh, thống kê kích thước, resize, merge các file nhãn.

## Ghi chú / Troubleshooting

- Nếu Streamlit báo lỗi không tìm thấy weights: đặt file weights vào đường dẫn `checkpoints/<model>/best.pt` hoặc sửa đường dẫn trong `app.py`.
- Nếu dùng GPU, đảm bảo `torch` cài phù hợp với CUDA phiên bản hệ thống.
- Một số hàm đọc CSV sẽ lọc các file không tồn tại trong thư mục ảnh — đảm bảo `filepath` trong `test.csv` tương ứng với tên file trong `img_dir`.

## Đóng góp

Mọi contribution (issue, PR) đều hoan nghênh. Trước khi đóng góp, vui lòng:

- Mô tả rõ mục tiêu (bugfix / feature).
- Nếu thêm model mới, cập nhật `evaluate_all.py` / `ensemble_score.py` để include model.

## Liên hệ

Nếu cần trợ giúp thêm hoặc muốn mình điều chỉnh README/ scripts, reply trực tiếp cho tôi.

---

README này là bản tóm tắt dựa trên mã nguồn hiện có trong thư mục `PAD/`. Nếu bạn muốn, tôi có thể:

- Bổ sung hướng dẫn từng bước cho từng script cụ thể (ví dụ: các tham số CLI của `train_vit.py`).
- Thêm ví dụ cấu hình `config/efficientnet.yaml` và giải thích các trường.

Hãy cho biết bạn muốn mở rộng phần nào nữa.
