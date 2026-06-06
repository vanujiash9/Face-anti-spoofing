import os
import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import sys

# Setup Path
sys.path.append(os.getcwd())

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
OUTPUT_DIR = "results/advanced_paper_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_DIR = "data/data_split/test"

# Đường dẫn chính xác tới các file weights của bạn
MODELS_CONFIG = {
    "ConvNeXt": {
        "class_name": "ConvNextBinary", 
        "path": "checkpoints/convnext/bestmodel_Convnext", 
        "size": 224
    },
    "EfficientNet": {
        "class_name": "EfficientNetBinary", 
        "path": "checkpoints/efficientnet/best.pt", 
        "size": 260
    },
    "ViT": {
        "class_name": "ViTBinary", 
        "path": "checkpoints/ViT/bestmodel_ViT", 
        "size": 224
    }
}

SPOOF_CATEGORIES = {
    'Deepfake': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter'],
    'Mask': ['silicon', 'mask', 'latex', 'UpperBodyMask', 'RegionMask', '3D_Mask'],
    'Print': ['Poster', 'Photo', 'A4', 'Print'],
    'Replay': ['Phone', 'PC', 'Pad', 'Screen', 'Replay']
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= ĐỊNH NGHĨA KIẾN TRÚC (TẮT PRETRAINED) =================

class EfficientNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Linear(self.backbone.num_features, 512), nn.BatchNorm1d(512),
                                  nn.SiLU(), nn.Dropout(0.5), nn.Linear(512, 1))
    def forward(self, x): return self.head(self.backbone(x)).squeeze(1)

class ConvNextBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.LayerNorm(self.backbone.num_features, eps=1e-6), nn.Flatten(1),
                                  nn.Dropout(0.5), nn.Linear(self.backbone.num_features, 1))
    def forward(self, x): return self.head(self.backbone(x)).squeeze(1)

class ViTBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.backbone.head = nn.Sequential(nn.Linear(self.backbone.head.in_features, 512),
                                           nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.5), nn.Linear(512, 1))
    def forward(self, x): return self.backbone(x).squeeze(1)

# ================= DATASET UTILS =================

class DetailedTestDataset(Dataset):
    def __init__(self, root_dir, size):
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
                    cat = "Live" if label == 0 else self.get_cat(f)
                    self.samples.append((os.path.join(path, f), label, cat))
    
    def get_cat(self, fname):
        for c, kws in SPOOF_CATEGORIES.items():
            if any(kw.lower() in fname.lower() for kw in kws): return c
        return "Other"

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l, c = self.samples[idx]
        return self.transform(Image.open(p).convert('RGB')), l, c, p

# ================= CORE FUNCTIONS =================

def analyze_model(name, cfg):
    print(f"\n--- Đang đánh giá chuyên sâu: {name} ---")
    
    # 1. Khởi tạo model sạch
    if name == "ConvNeXt": model = ConvNextBinary()
    elif name == "EfficientNet": model = EfficientNetBinary()
    else: model = ViTBinary()
    
    # 2. Nạp weights của bạn
    state_dict = torch.load(cfg['path'], map_location=device, weights_only=True)
    # Xử lý prefix 'module.' nếu train đa GPU
    clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model.to(device)
    model.eval()

    # 3. Load Data
    dataset = DetailedTestDataset(TEST_DIR, cfg['size'])
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    probs, labels, cats, paths = [], [], [], []
    with torch.no_grad():
        for imgs, lbls, categories, fpaths in tqdm(loader, desc=name):
            out = model(imgs.to(device))
            probs.extend(torch.sigmoid(out).cpu().numpy())
            labels.extend(lbls.numpy())
            cats.extend(categories)
            paths.extend(fpaths)
    
    df = pd.DataFrame({"path": paths, "label": labels, "category": cats, "prob": probs})
    
    # --- THỰC HIỆN CÁC YÊU CẦU CỦA GIẢNG VIÊN ---

    # A. Vẽ Histogram xác suất
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['label']==0]['prob'], color="green", label="Live", kde=True, bins=50, alpha=0.4)
    sns.histplot(df[df['label']==1]['prob'], color="red", label="Spoof", kde=True, bins=50, alpha=0.4)
    plt.title(f"Probability Distribution - {name}")
    plt.xlabel("Probability (Sigmoid)"); plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_histogram.png"), dpi=300)
    plt.close()

    # B. Vòng lặp Threshold (Bước nhảy 0.1)
    thresholds = np.arange(0, 1.1, 0.1)
    thresh_results = []
    for t in thresholds:
        preds = (df['prob'] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(df['label'], preds, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        thresh_results.append({
            "Threshold": round(t, 1), "TPR": tpr, "FPR": fpr, 
            "HTER": ((1-tpr) + fpr)/2, "Acc": (tp+tn)/len(df)
        })
    pd.DataFrame(thresh_results).to_csv(os.path.join(OUTPUT_DIR, f"{name}_threshold_analysis.csv"), index=False)

    # C. Lưu danh sách mẫu sai (Tại ngưỡng 0.5)
    df[(df['label'] == 1) & (df['prob'] < 0.5)].to_csv(os.path.join(OUTPUT_DIR, f"{name}_failed_spoof.csv"), index=False)
    df[(df['label'] == 0) & (df['prob'] >= 0.5)].to_csv(os.path.join(OUTPUT_DIR, f"{name}_failed_live.csv"), index=False)

    return df

def main():
    all_dfs = {}
    for name, cfg in MODELS_CONFIG.items():
        if os.path.exists(cfg['path']):
            all_dfs[name] = analyze_model(name, cfg)

    # D. So sánh độ chính xác theo 4 nhóm Spoof
    comparison = []
    for name, df in all_dfs.items():
        for cat in SPOOF_CATEGORIES.keys():
            sub = df[df['category'] == cat]
            if not sub.empty:
                acc = (sub['prob'] >= 0.5).mean()
                comparison.append({"Model": name, "Category": cat, "Accuracy": acc})
    
    if comparison:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=pd.DataFrame(comparison), x="Category", y="Accuracy", hue="Model")
        plt.ylim(0.9, 1.01); plt.title("Accuracy per Spoof Attack Category"); plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "overall_attack_comparison.png"), dpi=300)
    
    print(f"\n HOÀN TẤT! Kết quả tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()