import os
import cv2
import csv
import torch
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

CELEBA_DIR = r"dataset\CelebA spoof"
FFPP_DIR = r"dataset\FaceForensics++\fake"
DST_DIR = "SpoofDataset"
MIN_SIZE = 256
TOTAL_CELEBA_PER_FOLDER = 4002
TOTAL_FFPP_SUBFOLDER = 400

CSV_INDEX = os.path.join(DST_DIR, "dataset_index.csv")
CSV_MAP = os.path.join(DST_DIR, "dataset_original_map.csv")

os.makedirs(DST_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def laplacian_score_gpu(img):
    img_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).float().to(device)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]], dtype=torch.float32, device=device).unsqueeze(0)
    lap = torch.nn.functional.conv2d(img_tensor, kernel, padding=1)
    return torch.var(lap).item()

def laplacian_score(img):
    if device.type == 'cuda':
        return laplacian_score_gpu(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_best_image_paths(folder_path, total_per_folder):
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
    scored = []
    for img_file in imgs:
        path = os.path.join(folder_path, img_file)
        img = cv2.imread(path)
        if img is None or min(img.shape[:2]) < MIN_SIZE:
            continue
        score = laplacian_score(img)
        scored.append((score, img_file))
        del img
    scored.sort(reverse=True, key=lambda x:x[0])
    return [f for _,f in scored[:total_per_folder]]

def process_celeba_folder(folder_path, start_idx):
    spoof_type = os.path.basename(folder_path)
    best_files = get_best_image_paths(folder_path, TOTAL_CELEBA_PER_FOLDER)
    results = []
    idx = start_idx
    for img_file in best_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        new_name = f"CelebA{idx:06d}_1_{spoof_type}.jpg"
        cv2.imwrite(os.path.join(DST_DIR, new_name), img)
        del img
        results.append((new_name, 1, spoof_type, img_path))
        idx += 1
        if idx % 500 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Đã xử lý ~{idx} file CelebA")
    return results, idx

def gather_ffpp_subfolders():
    subfolders = []
    if not os.path.exists(FFPP_DIR):
        return subfolders
    for spoof_type in os.listdir(FFPP_DIR):
        folder_path = os.path.join(FFPP_DIR, spoof_type)
        if not os.path.isdir(folder_path):
            continue
        for sub in os.listdir(folder_path):
            sub_path = os.path.join(folder_path, sub)
            if os.path.isdir(sub_path):
                subfolders.append((spoof_type, sub_path))
    return subfolders[:TOTAL_FFPP_SUBFOLDER]

def get_best_image_from_folder(folder_path):
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
    best_score = -1
    best_file = None
    for img_file in img_files:
        path = os.path.join(folder_path, img_file)
        img = cv2.imread(path)
        if img is None or min(img.shape[:2])<MIN_SIZE:
            continue
        score = laplacian_score(img)
        if score>best_score:
            best_score = score
            best_file = img_file
        del img
    return best_file

def process_ffpp_folder(args):
    spoof_type, folder_path, idx = args
    best_file = get_best_image_from_folder(folder_path)
    if best_file is None:
        return [], idx
    img_path = os.path.join(folder_path, best_file)
    img = cv2.imread(img_path)
    new_name = f"FFPP{idx:06d}_1_{spoof_type}.jpg"
    cv2.imwrite(os.path.join(DST_DIR, new_name), img)
    del img
    return [(new_name, 1, spoof_type, img_path)], idx+1

def main():
    start_time = datetime.now()
    results = []
    idx = 1

    # CelebA
    celeba_folders = [os.path.join(CELEBA_DIR,f) for f in os.listdir(CELEBA_DIR) if os.path.isdir(os.path.join(CELEBA_DIR,f))]
    for folder in celeba_folders:
        r, idx = process_celeba_folder(folder, idx)
        results.extend(r)

    # FFPP
    ffpp_subfolders = gather_ffpp_subfolders()
    args_list = [(spf, subf, idx+i) for i,(spf,subf) in enumerate(ffpp_subfolders)]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_ffpp_folder, args) for args in args_list]
        for f in as_completed(futures):
            r,_ = f.result()
            results.extend(r)

    # CSV
    with open(CSV_INDEX,"w",newline="",encoding="utf-8") as f_idx,\
         open(CSV_MAP,"w",newline="",encoding="utf-8") as f_map:
        writer_idx = csv.writer(f_idx)
        writer_map = csv.writer(f_map)
        writer_idx.writerow(["path","label","type"])
        writer_map.writerow(["original_path","new_name"])
        for new_name,label,spoof,orig_path in results:
            writer_idx.writerow([new_name,label,spoof])
            writer_map.writerow([orig_path,new_name])

    print("------ XỬ LÝ HOÀN THÀNH ------")
    print(f"Tổng số file đã xử lý: {len(results)}")
    print(f"CSV Index: {CSV_INDEX}")
    print(f"CSV Original Map: {CSV_MAP}")
    print(f"Tổng thời gian: {datetime.now()-start_time}")

if __name__=="__main__":
    main()
