import os
import cv2
import torch
from datetime import datetime

SRC_DIR = r"dataset\FaceForencis++\fake"
DST_DIR = "SpoofDataset"
MIN_SIZE = 256
MAX_SUBFOLDER = 400  # chỉ lấy 400 folder đầu tiên

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Hàm tính Laplacian score trên GPU
def laplacian_score_gpu(img):
    img_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).float().to(device)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # NCHW
    kernel = torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype=torch.float32, device=device).unsqueeze(0)
    lap = torch.nn.functional.conv2d(img_tensor, kernel, padding=1)
    return torch.var(lap).item()

def laplacian_score(img):
    if device.type == 'cuda':
        return laplacian_score_gpu(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Lấy ảnh tốt nhất trong 1 folder
def get_best_image(folder_path):
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    best_score = -1
    best_img_file = None
    best_img = None
    for img_file in img_files:
        path = os.path.join(folder_path, img_file)
        img = cv2.imread(path)
        if img is None or min(img.shape[:2]) < MIN_SIZE:
            continue
        score = laplacian_score(img)
        if score > best_score:
            best_score = score
            best_img_file = img_file
            best_img = img
    return best_img_file, best_img

def main():
    start_time = datetime.now()
    os.makedirs(DST_DIR, exist_ok=True)

    # Chỉ lấy 5 folder chính
    spoof_types = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

    idx = 1
    results = []

    for spoof_type in spoof_types:
        type_path = os.path.join(SRC_DIR, spoof_type)
        if not os.path.exists(type_path):
            continue

        # Lấy 400 folder con đầu tiên
        subfolders = [os.path.join(type_path, d) for d in sorted(os.listdir(type_path)) if os.path.isdir(os.path.join(type_path, d))]
        subfolders = subfolders[:MAX_SUBFOLDER]

        for sub_path in subfolders:
            best_file, best_img = get_best_image(sub_path)
            if best_img is None:
                continue
            new_name = f"FFPP{idx:06d}_1_{spoof_type}.jpg"
            cv2.imwrite(os.path.join(DST_DIR, new_name), best_img)
            results.append((new_name, 1, spoof_type, os.path.join(sub_path, best_file)))
            idx += 1
            if idx % 50 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Đã xử lý ~{idx} file FFPP")

    print("------ XỬ LÝ HOÀN THÀNH ------")
    print(f"Tổng số file đã xử lý FFPP: {len(results)}")
    print(f"Tổng thời gian: {datetime.now() - start_time}")

if __name__ == "__main__":
    main()
