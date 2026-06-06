import os
from PIL import Image
import csv
from datetime import datetime, timedelta
from facenet_pytorch import MTCNN
import torch

SRC_DIRS = ['dataset/FFHQ', 'dataset/VGG/train']
DST_DIR = 'clean_live_sample'
CSV_OUTPUT = 'attack_tables_full/live_sample.csv'
MAX_FILES = {'dataset/FFHQ': 20000, 'dataset/VGG/train': 10000}
MIN_SIZE = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)

os.makedirs(DST_DIR, exist_ok=True)

def process_file(src_root, rel_path, idx):
    src_path = os.path.join(src_root, rel_path)
    if not os.path.exists(src_path):
        return None
    try:
        img = Image.open(src_path).convert('RGB')
        if min(img.size) < MIN_SIZE:
            return None
        face = mtcnn(img)
        if face is None:
            return None
    except:
        return None

    if 'FFHQ' in src_root:
        prefix = 'ffhq'
    else:
        prefix = 'vgg'

    new_filename = f"{prefix}_{idx:06d}_live.jpg"
    dst_path = os.path.join(DST_DIR, new_filename)
    img.save(dst_path)

    return {'path': new_filename, 'label': 1, 'type': 'live'}

file_list = []
for src_root in SRC_DIRS:
    count = 0
    for root, _, files in os.walk(src_root):
        for fn in files:
            rel_path = os.path.relpath(os.path.join(root, fn), src_root)
            file_list.append((src_root, rel_path))
            count += 1
            if count >= MAX_FILES[src_root]:
                break
        if count >= MAX_FILES[src_root]:
            break

start_time = datetime.now()
total_files = len(file_list)
results = []
passed_files = 0

for idx, (src_root, rel_path) in enumerate(file_list, 1):
    res = process_file(src_root, rel_path, idx)
    if res:
        results.append(res)
        passed_files += 1

    if idx % 5000 == 0 or idx == total_files:
        elapsed = datetime.now() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (total_files - idx)
        est_done = datetime.now() + remaining
        current_folder = src_root.split('/')[-1]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đã xử lý: {idx}/{total_files}, qua lọc: {passed_files}, folder: {current_folder}, dự kiến hoàn thành: {est_done.strftime('%H:%M:%S')}")

with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['path','label','type'])
    writer.writeheader()
    writer.writerows(results)

end_time = datetime.now()
print("------ XỬ LÝ HOÀN THÀNH ------")
print(f"Tổng file duyệt: {total_files}")
print(f"File lọc xong (có mặt): {passed_files}")
print(f"CSV đã lưu: {CSV_OUTPUT}")
print(f"Thời gian xử lý: {end_time - start_time}")
