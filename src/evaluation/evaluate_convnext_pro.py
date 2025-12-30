import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm

# Setup Path
import sys
sys.path.append(os.getcwd())
from src.models.build_model.convnext import ConvNextBinary

# ================= CẤU HÌNH =================
MODEL_NAME = "ConvNeXt"
CHECKPOINT_PATH = "checkpoints/convnext/best.pt" # Duong dan file ban vua train xong
LOG_PATH = "results/convnext/training_log.csv"
TEST_DIR = "data/data_split/test"
OUTPUT_DIR = "results/convnext_final_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dinh nghia cac nhom tan cong de phan tich sâu
SPOOF_CATEGORIES = {
    'Deepfake': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter'],
    'Mask': ['silicon', 'mask', 'latex', 'UpperBodyMask', 'RegionMask', '3D_Mask'],
    'Print': ['Poster', 'Photo', 'A4', 'Print'],
    'Replay': ['Phone', 'PC', 'Pad', 'Screen', 'Replay']
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM DATASET DE LAY THONG TIN CHI TIET ---
class DetailedFASDataset(Dataset):
    def __init__(self, root_dir, size=224):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.samples = []
        for label_folder in ['0_live', '1_spoof']:
            path = os.path.join(root_dir, label_folder)
            label = 0 if label_folder == '0_live' else 1
            for f in os.listdir(path):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    full_p = os.path.join(path, f)
                    # Phan loai tan cong dua tren ten file
                    cat = "Live" if label == 0 else self._get_attack_type(f)
                    self.samples.append((full_p, label, cat))
    
    def _get_attack_type(self, fname):
        for k, v in SPOOF_CATEGORIES.items():
            if any(x.lower() in fname.lower() for x in v): return k
        return "Other"

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l, c = self.samples[idx]
        return self.transform(Image.open(p).convert('RGB')), l, c, p

# --- CAC HAM TINH TOAN METRICS ---
def get_fas_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(float)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    apcer = fp / (fp + tn) if (fp + tn) > 0 else 0 # FAR
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0 # FRR
    return apcer, bpcer, (apcer + bpcer) / 2

def main():
    print(f"--- BAT DAU DANH GIA CHUYEN SAU MO HINH {MODEL_NAME.upper()} ---")
    
    # 1. Load Model & Weights
    model = ConvNextBinary().to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    # Fix DataParallel key prefix 'module.'
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    # 2. Load Data
    dataset = DetailedFASDataset(TEST_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 3. Inference
    all_probs, all_labels, all_cats, all_paths = [], [], [], []
    with torch.no_grad():
        for imgs, labels, cats, paths in tqdm(loader, desc="Inference"):
            out = model(imgs.to(device))
            all_probs.extend(torch.sigmoid(out).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_cats.extend(cats)
            all_paths.extend(paths)

    df = pd.DataFrame({
        "path": all_paths, "label": all_labels, 
        "category": all_cats, "prob": all_probs
    })

    # --- TRUC QUAN HOA 1: XU HUONG TRAINING (Dung file Log) ---
    if os.path.exists(LOG_PATH):
        print(">> Dang ve bieu do xu huong huan luyen...")
        log_df = pd.read_csv(LOG_PATH)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss')
        plt.plot(log_df['epoch'], log_df['val_loss'], label='Val Loss', linestyle='--')
        plt.title("Loss Curve"); plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(log_df['epoch'], log_df['hter'], label='Val HTER', color='orange')
        plt.title("HTER Evolution"); plt.legend()
        plt.savefig(f"{OUTPUT_DIR}/1_training_trends.png", dpi=300)

    # --- TRUC QUAN HOA 2: HISTOGRAM XAC SUAT ---
    print(">> Dang ve bieu do phan bo xac suat...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['label']==0]['prob'], color="green", label="Live", kde=True, bins=50)
    sns.histplot(df[df['label']==1]['prob'], color="red", label="Spoof", kde=True, bins=50)
    plt.title(f"Probability Distribution - {MODEL_NAME}")
    plt.legend(); plt.savefig(f"{OUTPUT_DIR}/2_probability_histogram.png", dpi=300)

    # --- TRUC QUAN HOA 3: ROC CURVE ---
    fpr_list, tpr_list, _ = roc_curve(df['label'], df['prob'])
    roc_auc = auc(fpr_list, tpr_list)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_list, tpr_list, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate (APCER)'); plt.ylabel('True Positive Rate (1-BPCER)')
    plt.title('Receiver Operating Characteristic'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/3_roc_curve.png", dpi=300)

    # --- TRUC QUAN HOA 4: HTER THEO LOAI TAN CONG ---
    print(">> Dang phan tich theo tung nhom tan cong...")
    attack_stats = []
    live_df = df[df['label'] == 0]
    for cat in SPOOF_CATEGORIES.keys():
        spoof_df = df[df['category'] == cat]
        if len(spoof_df) == 0: continue
        apcer, bpcer, hter = get_fas_metrics(
            np.concatenate([live_df['label'], spoof_df['label']]),
            np.concatenate([live_df['prob'], spoof_df['prob']])
        )
        attack_stats.append({"Attack": cat, "HTER": hter, "Samples": len(spoof_df)})
    
    df_attack = pd.DataFrame(attack_stats)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_attack, x='Attack', y='HTER', palette='magma')
    plt.title("HTER per Attack Category"); plt.savefig(f"{OUTPUT_DIR}/4_attack_analysis.png", dpi=300)

    # --- TRUC QUAN HOA 5: DA NGUONG (Step 0.1) ---
    thresholds = np.arange(0, 1.1, 0.1)
    thresh_report = []
    for t in thresholds:
        apcer, bpcer, hter = get_fas_metrics(df['label'].values, df['prob'].values, threshold=t)
        thresh_report.append({"Thresh": round(t, 1), "APCER": apcer, "BPCER": bpcer, "HTER": hter})
    
    df_thresh = pd.DataFrame(thresh_report)
    df_thresh.to_csv(f"{OUTPUT_DIR}/5_threshold_sensitivity.csv", index=False)

    # --- TRUC QUAN HOA 6: TPR @ LOW FPR ---
    # Tim TPR tai FPR = 1%
    tpr_at_1fpr = np.interp(0.01, fpr_list, tpr_list)
    print(f"\n--- KET QUA CUOI CUNG {MODEL_NAME} ---")
    print(f"TPR @ 1% FPR: {tpr_at_1fpr:.4%}")
    print(f"HTER (at 0.5): {df_thresh[df_thresh['Thresh']==0.5]['HTER'].values[0]:.4%}")
    print(f"AUC Score: {roc_auc:.4f}")

    # --- TRUC QUAN HOA 7: MA TRAN NHAM LAN CHUAN HOA ---
    cm = confusion_matrix(df['label'], (df['prob'] >= 0.5).astype(int))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=['Live', 'Spoof'], yticklabels=['Live', 'Spoof'])
    plt.title("Normalized Confusion Matrix"); plt.savefig(f"{OUTPUT_DIR}/6_confusion_matrix.png", dpi=300)

    print(f"\n HOAN TAT! Tat ca bao cao da luu tai: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()