import os
import torch
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
import imagehash 

# ================= CẤU HÌNH KHẮT KHE =================
ROOT_DIR = os.path.join('data', 'data_process', 'face_forest')
OUTPUT_DIR = os.path.join('data', 'data_process', 'cropped_faces')

# Nếu chạy CPU thì giảm Batch Size xuống để tránh treo máy
BATCH_SIZE = 8 if torch.cuda.is_available() else 4
MIN_INPUT_SIZE = 224     
SCALE_FACTOR = 1.4       

# --- BỘ LỌC CHẤT LƯỢNG ---
MIN_FACE_PIXELS = 80     
BLUR_THRESHOLD = 50      
HASH_DIFF_CUTOFF = 5     

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 DEVICE: {device}")
print(f"🛡️ MODE: Deep Cleaning (Auto-Fallback for Mixed Sizes)")

mtcnn = MTCNN(
    keep_all=True, 
    device=device, 
    margin=0, 
    min_face_size=MIN_FACE_PIXELS,
    post_process=False
)

# ================= CÁC HÀM XỬ LÝ =================

def standardize_filename(filename, dataset_name):
    name_body, ext = os.path.splitext(filename)
    label = 'Check'
    spoof_type = 'Unknown'
    raw_id = name_body
    parts = name_body.split('_')

    if '_0_' in filename or filename.endswith('_0' + ext):
        label = '0'; spoof_type = 'Real'
        temp_parts = name_body.rsplit('_0_', 1)
        if len(temp_parts) > 0: raw_id = temp_parts[0]
    elif '_1_' in filename:
        label = '1'
        reversed_parts = parts[::-1]
        try:
            if '1' in reversed_parts:
                idx_rev = reversed_parts.index('1')
                idx_real = len(parts) - 1 - idx_rev
                raw_id = "_".join(parts[:idx_real])
                spoof_type = "_".join(parts[idx_real + 1:])
                if not spoof_type: spoof_type = "Spoof"
        except: pass
    
    if raw_id.lower().startswith(dataset_name.lower()):
        raw_id = raw_id[len(dataset_name):].strip('_')
    if not raw_id: raw_id = "id"
    spoof_type = spoof_type.replace('/', '-').replace('\\', '-')

    return f"{dataset_name}_{raw_id}_{label}_{spoof_type}{ext}"

def is_blurry(image_pil, threshold=BLUR_THRESHOLD):
    img_np = np.array(image_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, score

def get_loose_square_box(box, img_w, img_h, scale=1.3):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    max_side = max(w, h)
    new_side = int(max_side * scale)
    half_side = new_side // 2
    
    nx1 = max(0, cx - half_side)
    ny1 = max(0, cy - half_side)
    nx2 = min(img_w, cx + half_side)
    ny2 = min(img_h, cy + half_side)
    return (nx1, ny1, nx2, ny2)

seen_hashes = {} 

def process_batch(batch_files, batch_info):
    batch_imgs = []
    valid_indices = []
    
    # 1. Load ảnh
    for i, path in enumerate(batch_files):
        try:
            img = Image.open(path).convert('RGB')
            batch_imgs.append(img)
            valid_indices.append(i)
        except: continue

    if not batch_imgs: return 0, 0

    boxes_list = []
    probs_list = []

    # 2. Detect (ĐÃ SỬA LỖI Ở ĐÂY)
    try:
        # Cố gắng chạy batch nhanh
        boxes_list, probs_list = mtcnn.detect(batch_imgs)
    except Exception as e:
        # Nếu lỗi (do ảnh không đều size HOẶC lỗi GPU), chuyển sang chạy từng ảnh
        # "Exception" bắt tất cả các loại lỗi bao gồm lỗi "equal-dimension"
        boxes_list = []
        probs_list = []
        for img in batch_imgs:
            try:
                b, p = mtcnn.detect(img)
                boxes_list.append(b)
                probs_list.append(p)
            except:
                boxes_list.append(None)
                probs_list.append(None)

    # 3. Filter & Save
    saved_count = 0
    filtered_count = 0 
    
    for idx, boxes in enumerate(boxes_list):
        if boxes is None: continue
        probs = probs_list[idx]
        if probs is None: continue

        orig_idx = valid_indices[idx]
        ds_name, output_folder = batch_info[orig_idx]
        src_path = batch_files[orig_idx]
        img = batch_imgs[idx]

        if ds_name not in seen_hashes: seen_hashes[ds_name] = []

        best_idx = np.argmax(probs)
        if probs[best_idx] < 0.95: 
            filtered_count += 1
            continue 

        box = boxes[best_idx]
        
        fw = box[2] - box[0]
        fh = box[3] - box[1]
        if fw < MIN_FACE_PIXELS or fh < MIN_FACE_PIXELS:
            filtered_count += 1
            continue

        try:
            crop_box = get_loose_square_box(box, img.size[0], img.size[1], scale=SCALE_FACTOR)
            face_img = img.crop(crop_box)
            
            is_bad, blur_score = is_blurry(face_img, threshold=BLUR_THRESHOLD)
            if is_bad:
                filtered_count += 1
                continue

            curr_hash = imagehash.phash(face_img)
            is_duplicate = False
            
            recent_hashes = seen_hashes[ds_name][-50:] 
            for h in recent_hashes:
                if (curr_hash - h) < HASH_DIFF_CUTOFF:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                filtered_count += 1
                continue
            
            seen_hashes[ds_name].append(curr_hash)
            
            w, h = face_img.size
            if w < MIN_INPUT_SIZE or h < MIN_INPUT_SIZE:
                face_img = face_img.resize((MIN_INPUT_SIZE, MIN_INPUT_SIZE), Image.BICUBIC)
            
            filename = os.path.basename(src_path)
            new_filename = standardize_filename(filename, ds_name)
            save_path = os.path.join(output_folder, new_filename)
            
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                face_img.save(save_path, quality=95)
                saved_count += 1
                
        except Exception:
            pass
            
    return saved_count, filtered_count

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"❌ Không tìm thấy: {ROOT_DIR}")
        return

    print("📂 Đang quét danh sách file...")
    all_tasks = []
    
    ds_folders = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    for ds_name in ds_folders:
        ds_in = os.path.join(ROOT_DIR, ds_name)
        ds_out = os.path.join(OUTPUT_DIR, ds_name)
        for root, _, files in os.walk(ds_in):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_tasks.append({
                        'path': os.path.join(root, f),
                        'ds': ds_name,
                        'out': ds_out
                    })
    
    total = len(all_tasks)
    print(f"📊 Tổng số ảnh thô: {total}")
    print("🚀 Bắt đầu DEEP CLEANING...")

    total_saved = 0
    total_filtered = 0

    with tqdm(total=total, unit="img") as pbar:
        for i in range(0, total, BATCH_SIZE):
            batch_items = all_tasks[i : i + BATCH_SIZE]
            b_files = [x['path'] for x in batch_items]
            b_info = [(x['ds'], x['out']) for x in batch_items]
            
            saved, filtered = process_batch(b_files, b_info)
            total_saved += saved
            total_filtered += filtered
            
            pbar.update(len(batch_items))
            pbar.set_postfix({'Saved': total_saved, 'Trash': total_filtered})
            
    print("=" * 60)
    print(f"✅ HOÀN TẤT! Đã lưu: {total_saved} ảnh sạch.")

if __name__ == "__main__":
    main()