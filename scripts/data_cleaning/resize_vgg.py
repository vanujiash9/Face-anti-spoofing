import os
from PIL import Image
import csv
from datetime import datetime, timedelta
from facenet_pytorch import MTCNN
import torch

SRC_DIR = 'dataset/VGG/train'
DST_DIR = 'clean_live_sample'
CSV_OUTPUT = 'attack_tables_full/live_sample.csv'

MAX_PER_FOLDER = 200    # mỗi folder lấy tối đa 200 ảnh
TOTAL_TARGET = 12000    # tổng số ảnh cần
MIN_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == 'cuda':
    print("GPU is available!")
    print("Device name:", torch.cuda.get_device_name(0))
    print("Total memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
else:
    print("GPU not available, using CPU.")

# --- MTCNN với GPU ---
mtcnn = MTCNN(keep_all=False, device=device)

os.makedirs(DST_DIR, exist_ok=True)

# --- Xử lý ảnh ---
results = []
idx_global = 1
start_time = datetime.now()

for folder in os.listdir(SRC_DIR):
    folder_path = os.path.join(SRC_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    imgs = imgs[:MAX_PER_FOLDER]
    
    for img_file in imgs:
        try:
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert("RGB")
            if min(img.size) < MIN_SIZE:
                continue
            face = mtcnn(img)
            if face is None:
                continue
            # lưu ảnh
            new_name = f"vgg_{idx_global:06d}_live.jpg"
            img.save(os.path.join(DST_DIR, new_name))
            results.append({"path": new_name, "label": 1, "type": "live"})
            idx_global += 1

        except:
            continue

        # --- Log mỗi 500 ảnh ---
        if idx_global % 500 == 0:
            elapsed = datetime.now() - start_time
            avg_time = elapsed.total_seconds() / idx_global
            remaining = avg_time * (TOTAL_TARGET - idx_global)
            est_done = datetime.now() + timedelta(seconds=remaining)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Đã xử lý: {idx_global}/{TOTAL_TARGET}, dự kiến hoàn thành: {est_done.strftime('%H:%M:%S')}")

        if idx_global > TOTAL_TARGET:
            break
    if idx_global > TOTAL_TARGET:
        break

# --- Lưu CSV ---
with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=['path', 'label', 'type'])
    writer.writeheader()
    writer.writerows(results)

end_time = datetime.now()
print("------ XỬ LÝ HOÀN THÀNH ------")
print(f"Tổng số file đã lưu: {idx_global-1}")
print(f"CSV đã lưu: {CSV_OUTPUT}")
print(f"Thời gian xử lý: {end_time - start_time}")
