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
import sys

# Setup Path
sys.path.append(os.getcwd())
from src.models.build_model.vit import ViTLoRA 
from src.training.utils import compute_metrics, get_sys_stats

# ================= CẤU HÌNH =================
MODEL_NAME = "ViT"
CHECKPOINT_PATH = "checkpoints/vit/best.pt" 
LOG_PATH = "results/vit/training_log.csv"
TEST_DIR = "data/data_split/test"
OUTPUT_DIR = "results/vit_final_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPOOF_CATEGORIES = {
    'Deepfake': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter'],
    'Mask': ['silicon', 'mask', 'latex', 'UpperBodyMask', 'RegionMask', '3D_Mask'],
    'Print': ['Poster', 'Photo', 'A4', 'Print'],
    'Replay': ['Phone', 'PC', 'Pad', 'Screen', 'Replay']
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            if not os.path.exists(path): continue
            for f in os.listdir(path):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    cat = "Live" if label == 0 else self._get_attack_type(f)
                    self.samples.append((os.path.join(path, f), label, cat))
    
    def _get_attack_type(self, fname):
        for k, v in SPOOF_CATEGORIES.items():
            if any(x.lower() in fname.lower() for x in v): return k
        return "Other"
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l, c = self.samples[idx]
        return self.transform(Image.open(p).convert('RGB')), l, c, p

def main():
    print(f"--- BAT DAU DANH GIA CHUYEN SAU ViT ---")
    model = ViTLoRA().to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    dataset = DetailedFASDataset(TEST_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_probs, all_labels, all_cats, all_paths = [], [], [], []
    with torch.no_grad():
        for imgs, labels, cats, paths in tqdm(loader, desc="Inference"):
            out = model(imgs.to(device))
            all_probs.extend(torch.sigmoid(out).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_cats.extend(cats)
            all_paths.extend(paths)

    df = pd.DataFrame({"path": all_paths, "label": all_labels, "category": all_cats, "prob": all_probs})

    # 1. Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['label']==0]['prob'], color="green", label="Live", kde=True, bins=50)
    sns.histplot(df[df['label']==1]['prob'], color="red", label="Spoof", kde=True, bins=50)
    plt.title(f"Probability Distribution - ViT"); plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/2_probability_histogram.png", dpi=300)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(df['label'], df['prob'])
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], 'k--'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/3_roc_curve.png", dpi=300)

    # 3. Attack Analysis
    attack_stats = []
    live_df = df[df['label'] == 0]
    for cat in SPOOF_CATEGORIES.keys():
        spoof_df = df[df['category'] == cat]
        if len(spoof_df) == 0: continue
        y_test = np.concatenate([live_df['label'], spoof_df['label']])
        y_prob = np.concatenate([live_df['prob'], spoof_df['prob']])
        y_pred = (y_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
        hter = ((fp/(fp+tn)) + (fn/(fn+tp))) / 2
        attack_stats.append({"Attack": cat, "HTER": hter})
    
    sns.barplot(data=pd.DataFrame(attack_stats), x='Attack', y='HTER')
    plt.title("HTER per Attack Type"); plt.savefig(f"{OUTPUT_DIR}/4_attack_analysis.png", dpi=300)

    # 4. Threshold CSV
    thresh_report = []
    for t in np.arange(0, 1.1, 0.1):
        y_p = (df['prob'] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(df['label'], y_p, labels=[0,1]).ravel()
        apcer, bpcer = fp/(fp+tn), fn/(fn+tp)
        thresh_report.append({"Thresh": round(t, 1), "APCER": apcer, "BPCER": bpcer, "HTER": (apcer+bpcer)/2})
    pd.DataFrame(thresh_report).to_csv(f"{OUTPUT_DIR}/5_threshold_sensitivity.csv", index=False)

    print(f"HTER (0.5): {thresh_report[5]['HTER']:.4%}")
    print(f" DONE. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
