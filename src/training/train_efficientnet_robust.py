import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from safetensors.torch import load_file

# Import tu data_loader_robust.py của bạn
from src.data.data_loader import build_loaders

# ==========================================
# 1. MODEL DEFINITION (EFFICIENTNET + HEAD)
# ==========================================

class FASEfficientNet(nn.Module):
    def __init__(self, weight_path):
        super().__init__()
        # 1. Load backbone khong pretrained tu timm
        self.backbone = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=0)
        
        # 2. Nap trọng số safetensors cục bộ
        state_dict = load_file(weight_path)
        self.backbone.load_state_dict(state_dict, strict=False)
        
        # 3. Head chuyên biệt cho FAS (Dropout cao 0.5)
        self.num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        # EfficientNet num_classes=0 tra ve (B, C) sau pooling
        return self.head(features).squeeze(1)

# ==========================================
# 2. UTILS
# ==========================================

def calculate_hter(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(float)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    apcer = fp / (fp + tn) if (fp + tn) > 0 else 0
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0
    return (apcer + bpcer) / 2, apcer, bpcer

def set_trainable(model, epoch):
    # Epoch 1-2: Freeze backbone, chi train head
    if epoch <= 2:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        # Epoch 3+: Mo khoa toan bo
        for param in model.parameters():
            param.requires_grad = True

# ==========================================
# 3. MAIN TRAINING
# ==========================================

def main():
    # --- CONFIG ---
    DATA_DIR = "data/dataset_final"
    WEIGHT_PATH = "saved_models/tf_efficientnetv2_b0/model.safetensors"
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16 # An toan cho RTX 3050 4GB/8GB
    EPOCHS = 20
    ALPHA = 0.1 # Label smoothing
    
    print(f"Device: {DEVICE} | Model: EfficientNet-V2-B0")
    
    # Load Data
    train_loader, val_loader = build_loaders(DATA_DIR, 224, BATCH_SIZE)
    
    # Load Model
    model = FASEfficientNet(WEIGHT_PATH).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    best_hter = 1.0

    print("--- BAT DAU HUAN LUYEN EFFICIENTNET ---")

    for epoch in range(EPOCHS):
        set_trainable(model, epoch + 1)
        
        # --- TRAIN ---
        model.train()
        running_t_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # Manual Label Smoothing (0 -> 0.05, 1 -> 0.95)
            smoothed_labels = labels * (1 - ALPHA) + 0.5 * ALPHA
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, smoothed_labels)
            loss.backward()
            optimizer.step()
            
            running_t_loss += loss.item()
            pbar.set_postfix({'T_Loss': f"{loss.item():.4f}"})

        # --- VAL ---
        model.eval()
        running_v_loss = 0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                running_v_loss += loss.item()
                
                all_probs.extend(torch.sigmoid(out).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Metrics
        avg_t_loss = running_t_loss / len(train_loader)
        avg_v_loss = running_v_loss / len(val_loader)
        hter, apcer, bpcer = calculate_hter(np.array(all_labels), np.array(all_probs))
        acc = (np.array(all_labels) == (np.array(all_probs) >= 0.5)).mean()
        
        scheduler.step()

        # Log to CSV
        history.append({
            "epoch": epoch+1, "t_loss": avg_t_loss, "v_loss": avg_v_loss, 
            "hter": hter, "acc": acc
        })
        pd.DataFrame(history).to_csv("results/efficientnet_robust_logs.csv", index=False)

        print(f"Ep {epoch+1} | T_Loss: {avg_t_loss:.4f} | V_Loss: {avg_v_loss:.4f} | Acc: {acc:.4f} | HTER: {hter:.4f}")

        if hter < best_hter:
            best_hter = hter
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/efficientnet_robust_best.pt")
            print(">>> Saved Best Model")

if __name__ == "__main__":
    main()