import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from tqdm import tqdm
import sys

# Setup Path
sys.path.append(os.getcwd())
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTLoRA
from src.data.data_loader import build_loaders

# ================= CẤU HÌNH HỆ THỐNG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results/paper_final_figures_v2"
DATA_DIR = "data/data_split"

MODELS_CONFIG = {
    "ConvNeXt": {
        "class": ConvNextBinary, "path": "checkpoints/convnext/best.pt", 
        "size": 224, "color": "#1f77b4", "log": "results/convnext/training_log.csv"
    },
    "EfficientNet": {
        "class": EfficientNetBinary, "path": "checkpoints/efficientnet/best.pt", 
        "size": 260, "color": "#ff7f0e", "log": "results/efficientnet/training_log.csv"
    },
    "ViT": {
        "class": ViTLoRA, "path": "checkpoints/vit/best.pt", 
        "size": 224, "color": "#2ca02c", "log": "results/vit/training_log.csv"
    }
}

SPOOF_GROUPS = {
    'Deepfake': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
    'Mask': ['silicon', 'mask', 'latex', 'UpperBodyMask', 'RegionMask', '3D_Mask'],
    'Print': ['Poster', 'Photo', 'A4'],
    'Replay': ['Phone', 'PC', 'Pad']
}

# Tắt warning và set style
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
sns.set_style("whitegrid")

# ================= HÀM HỖ TRỢ =================

def load_model_safely(name, cfg):
    print(f">>> Loading model {name}...")
    if not os.path.exists(cfg['path']): return None
    model = cfg['class']().to(DEVICE)
    state = torch.load(cfg['path'], map_location=DEVICE, weights_only=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()
    return model

def get_predictions(model, loader):
    probs, labels, fnames = [], [], []
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(tqdm(loader, leave=False, desc="Inference")):
            out = torch.sigmoid(model(imgs.to(DEVICE)))
            probs.extend(out.cpu().numpy())
            labels.extend(lbls.numpy())
            start = i * loader.batch_size
            fnames.extend([os.path.basename(x[0]) for x in loader.dataset.samples[start:start+len(lbls)]])
    return np.array(probs), np.array(labels), fnames

# ================= CÁC MODULE TRỰC QUAN HÓA =================

# 1. CHI TIẾT LOSS & HTER TỪNG MÔ HÌNH (Dùng để soi Overfitting)
def plot_individual_learning_curves():
    print(">>> Fig 1: Individual Learning Curves (Overfitting Check)...")
    for name, cfg in MODELS_CONFIG.items():
        if os.path.exists(cfg['log']):
            df = pd.read_csv(cfg['log'])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Trái: Loss
            ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', lw=2)
            ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red', linestyle='--', lw=2)
            ax1.set_title(f"{name}: Loss Analysis")
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
            
            # Phải: HTER
            hter_col = 'val_hter' if 'val_hter' in df.columns else 'hter'
            ax2.plot(df['epoch'], df[hter_col], label='Val HTER', color='green', lw=2)
            ax2.set_title(f"{name}: Error Rate (HTER)")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("HTER"); ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/Fig1_Learning_{name}.png", dpi=300)
            plt.close()

# 2. DATASET STATISTICS (Hàn lâm hơn)
def plot_dataset_stats():
    print(">>> Fig 2: Advanced Dataset Stats...")
    data = {'Category': ['Live', 'Mask', 'Deepfake', 'Print', 'Replay'],
            'Count': [36104, 3915, 2959, 2750, 2260]}
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Category', y='Count', palette="Blues_d")
    plt.title("Detailed Dataset Distribution", fontsize=14, fontweight='bold')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.savefig(f"{OUTPUT_DIR}/Fig2_Dataset_Stats.png", dpi=300)
    plt.close()

# 3. FIX T-SNE: HIỂN THỊ CẢ 2 MÀU (Live vs Spoof)
def plot_tsne_fixed(all_data):
    print(">>> Fig 3: Fixed t-SNE (Feature Separation)...")
    name = "ConvNeXt" # Hoặc chọn model bạn muốn
    m = load_model_safely(name, MODELS_CONFIG[name])
    _, loader = build_loaders(DATA_DIR, 224, 32)
    
    feats, lbls = [], []
    # Lấy mẫu cân bằng
    with torch.no_grad():
        for imgs, l in loader:
            f = m.backbone(imgs.to(DEVICE))
            if len(f.shape) == 4: f = f.mean(dim=[2,3])
            feats.append(f.cpu().numpy()); lbls.append(l.numpy())
            if len(np.concatenate(lbls)) > 1500: break

    feats = np.concatenate(feats)
    lbls = np.concatenate(lbls)
    
    x_embed = TSNE(n_components=2, perplexity=30, max_iter=1000).fit_transform(feats)
    plt.figure(figsize=(10, 8))
    # Dùng nhãn 0=Live (Green), 1=Spoof (Red)
    scatter = plt.scatter(x_embed[:,0], x_embed[:,1], c=lbls, cmap='RdYlGn_r', alpha=0.6, edgecolors='w')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Live', 'Spoof'])
    plt.title(f"Feature Space Visualization (t-SNE) - {name}", fontsize=14, fontweight='bold')
    plt.savefig(f"{OUTPUT_DIR}/Fig3_tSNE_Fixed.png", dpi=300)
    plt.close()

# 4. MA TRẬN NHẦM LẪN CHUẨN (Counts + %)
def plot_confusion_matrix_final(all_data):
    print(">>> Fig 4: Confusion Matrices (Counts & %)...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (name, data) in enumerate(all_data.items()):
        cm = confusion_matrix(data['labels'], (data['probs'] >= 0.5).astype(int))
        # Tạo label chứa cả số lượng và phần trăm
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=axes[i], cbar=False)
        axes[i].set_title(f"{name}")
        axes[i].set_xticklabels(['Pred Real', 'Pred Spoof']); axes[i].set_yticklabels(['Actual Real', 'Actual Spoof'])
    plt.savefig(f"{OUTPUT_DIR}/Fig4_Confusion_Matrices.png", dpi=300)
    plt.close()

# 5. GALLERY: ẢNH ĐÚNG VÀ ẢNH SAI (FAILURE ANALYSIS)
def plot_qualitative_analysis(all_data):
    print(">>> Fig 5: Qualitative Error Analysis...")
    name = "ConvNeXt" # Model tiêu biểu
    data = all_data[name]
    probs, labels, fnames = data['probs'], data['labels'], data['fnames']
    
    # Tìm mẫu sai nặng nhất
    failed_indices = np.where((labels == 1) & (probs < 0.2))[0] # Spoof lọt lưới
    if len(failed_indices) < 4: failed_indices = np.where(labels != (probs >= 0.5).astype(int))[0]
    
    # Chọn 4 mẫu sai và 4 mẫu đúng
    correct_indices = np.where(labels == (probs >= 0.5).astype(int))[0]
    
    selected_idx = list(np.random.choice(correct_indices, 4)) + list(np.random.choice(failed_indices, 4))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, idx in enumerate(selected_idx):
        ax = axes[i//4, i%4]
        img_path = os.path.join(DATA_DIR, "test", "1_spoof" if labels[idx]==1 else "0_live", fnames[idx])
        ax.imshow(Image.open(img_path))
        
        pred = "SPOOF" if probs[idx] >= 0.5 else "LIVE"
        gt = "SPOOF" if labels[idx] == 1 else "LIVE"
        color = "green" if (probs[idx] >= 0.5) == labels[idx] else "red"
        
        ax.set_title(f"GT: {gt}\nPred: {pred}\nScore: {probs[idx]:.4f}", color=color, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle("Qualitative Results: Correct Predictions vs Failure Cases", fontsize=18, y=0.98)
    plt.savefig(f"{OUTPUT_DIR}/Fig5_Qualitative_Analysis.png", dpi=300)
    plt.close()

# ================= MAIN EXECUTION =================

def main():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_data = {}
    for name, cfg in MODELS_CONFIG.items():
        m = load_model_safely(name, cfg)
        if m is None: continue
        _, loader = build_loaders(DATA_DIR, cfg['size'], 32)
        p, l, f = get_predictions(m, loader)
        all_data[name] = {'probs': p, 'labels': l, 'fnames': f}

    plot_individual_learning_curves()
    plot_dataset_stats()
    plot_tsne_fixed(all_data)
    plot_confusion_matrix_final(all_data)
    plot_qualitative_analysis(all_data)
    
    # Biểu đồ dải Threshold tổng hợp (Fig 6)
    plt.figure(figsize=(10, 6))
    for name, data in all_data.items():
        fpr, tpr, _ = roc_curve(data['labels'], data['probs'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.4f})", color=MODELS_CONFIG[name]['color'], lw=2)
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curves Comparison", fontsize=14, fontweight='bold')
    plt.legend(); plt.savefig(f"{OUTPUT_DIR}/Fig6_ROC_Comparison.png", dpi=300)
    plt.close()

    print(f"\nTHÀNH CÔNG! Đã xuất đầy đủ 10+ Figure báo cáo tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()