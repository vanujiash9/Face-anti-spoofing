import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm

# --- IMPORT CẤU TRÚC CỦA BẠN ---
from src.models.build_model.efficientnet import EfficientNetBinary
from src.data.data_loader import build_loaders

# ================= CẤU HÌNH =================
# SỬ DỤNG GPU 1 (VÌ GPU 0 ĐANG BẬN)
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "checkpoints/efficientnet/best.pt"
DATA_DIR = "data/data_split"
OUTPUT_DIR = "results/stress_test_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPOOF_CATEGORIES = {
    'Deepfake': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
    'Mask': ['silicon', 'mask', 'latex', 'UpperBodyMask', 'RegionMask', '3D_Mask'],
    'Print': ['Poster', 'Photo', 'A4', 'Print'],
    'Replay': ['Phone', 'PC', 'Pad', 'Screen', 'Replay']
}

def get_category(fname):
    for cat, keywords in SPOOF_CATEGORIES.items():
        if any(kw.lower() in fname.lower() for kw in keywords): return cat
    return 'Other'

def main():
    print(f"--- ĐANG THỰC HIỆN STRESS TEST TRÊN THIẾT BỊ: {device} ---")
    
    # 1. Load Model
    model = EfficientNetBinary().to(device)
    if not os.path.exists(CHECKPOINT):
        print(f"Lỗi: Không tìm thấy file {CHECKPOINT}")
        return

    state_dict = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    # 2. Load Data (Size 260 chuẩn EfficientNet)
    _, test_loader = build_loaders(DATA_DIR, 260, batch_size=32, num_workers=4)
    
    all_results = []
    print("Bước 1: Chạy Inference toàn bộ tập Test...")
    
    with torch.no_grad():
        # Lấy danh sách file thực tế từ dataset
        samples = test_loader.dataset.samples
        for i, (imgs, labels) in enumerate(tqdm(test_loader)):
            outputs = model(imgs.to(device))
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            batch_start = i * 32
            for j in range(len(labels)):
                full_path = samples[batch_start + j][0]
                fname = os.path.basename(full_path)
                all_results.append({
                    'path': full_path,
                    'label': int(labels[j]),
                    'prob': float(probs[j]),
                    'category': 'Live' if labels[j] == 0 else get_category(fname)
                })

    df = pd.DataFrame(all_results)

    # --- BÀI KIỂM TRA 1: PHÂN TÍCH NHÓM TẤN CÔNG ---
    print("\nBước 2: Phân tích HTER theo từng loại tấn công...")
    group_stats = []
    live_df = df[df['label'] == 0]
    
    for cat in SPOOF_CATEGORIES.keys():
        spoof_df = df[df['category'] == cat]
        if len(spoof_df) == 0: continue
        
        # Tính metrics cho nhóm spoof này tại ngưỡng 0.5
        fn = (spoof_df['prob'] < 0.5).sum()
        fp = (live_df['prob'] >= 0.5).sum()
        apcer = fp / len(live_df)
        bpcer = fn / len(spoof_df)
        hter = (apcer + bpcer) / 2
        
        group_stats.append({'Category': cat, 'HTER (%)': hter * 100, 'Count': len(spoof_df)})

    df_groups = pd.DataFrame(group_stats)
    print(df_groups.to_string(index=False))

    # --- BÀI KIỂM TRA 2: TRỰC QUAN HÓA Histogram ---
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="prob", hue="label", bins=50, kde=True, palette={0: "green", 1: "red"})
    plt.axvline(0.5, color='blue', linestyle='--', label='Threshold 0.5')
    plt.title("Phân bổ xác suất: Live (Green) vs Spoof (Red)")
    plt.xlabel("Xác suất dự đoán (0=Real, 1=Spoof)")
    plt.savefig(f"{OUTPUT_DIR}/probability_histogram.png", dpi=300)
    print(f"\n✅ Đã lưu Histogram tại: {OUTPUT_DIR}/probability_histogram.png")

    # --- BÀI KIỂM TRA 3: TPR @ 1% FPR (CHỈ SỐ NGHIỆM THU) ---
    fpr, tpr, _ = roc_curve(df['label'], df['prob'])
    idx = np.abs(fpr - 0.01).argmin()
    tpr_at_1fpr = tpr[idx]
    print(f"\n--- KẾT QUẢ KIỂM ĐỊNH TÍNH TỔNG QUÁT ---")
    print(f"TPR @ 1% FPR: {tpr_at_1fpr:.4f}")
    
    # --- BÀI KIỂM TRA 4: TRUY VẾT MẪU SAI ĐỂ BIỆN LUẬN ---
    # Lọc những ảnh Spoof mà AI đoán cực kỳ thấp (Sai nặng)
    critical_fails = df[(df['label'] == 1) & (df['prob'] < 0.2)].sort_values(by='prob')
    critical_fails.to_csv(f"{OUTPUT_DIR}/critical_spoof_failures.csv", index=False)
    print(f"Đã lưu danh sách {len(critical_fails)} mẫu giả mạo 'lọt lưới' nặng nhất để bạn soi ảnh.")

if __name__ == "__main__":
    main()