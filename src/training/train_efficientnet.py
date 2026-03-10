import os
import time
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.data.data_loader import build_loaders
from src.models.build_model.efficientnet import EfficientNetBinary
from src.training.utils import get_sys_stats, compute_metrics, save_training_log

def mixup_data(x, y, alpha=0.2):
    """Trộn dữ liệu để mô hình không học thuộc lòng từng ảnh đơn lẻ"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def main():
    # 1. Load Cấu hình riêng cho EfficientNet
    with open("config/efficientnet.yaml") as f: cfg = yaml.safe_load(f)
    device = torch.device(cfg['train']['device'])
    
    checkpoint_dir = cfg['train']['checkpoint_dir']
    results_dir = cfg['train']['results_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Khởi tạo Data Loader
    train_loader, val_loader = build_loaders(
        cfg['dataset']['img_dir'], 
        cfg['dataset']['input_size'], 
        cfg['dataset']['batch_size'],
        num_workers=cfg['dataset']['num_workers']
    )
    
    # 3. Khởi tạo Mô hình
    model = EfficientNetBinary().to(device)
    
    # CHIẾN THUẬT LR: Head học 1e-4, Backbone học 1e-5 (Differential LR)
    base_lr = float(cfg['train']['learning_rate'])
    optimizer = torch.optim.AdamW([
        {'params': model.head.parameters(), 'lr': base_lr},
        {'params': model.backbone.parameters(), 'lr': base_lr * 0.1}
    ], weight_decay=float(cfg['train']['weight_decay']))
    
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    
    history = []
    best_hter = 1.0
    patience_cnt = 0
    patience_limit = cfg['train']['patience']
    sm = cfg['train']['label_smoothing']

    print(f"--- STARTING EFFICIENTNET TRAINING | GPU: {device} ---")

    for epoch in range(cfg['train']['epochs']):
        start_time = time.time()
        
        # Progressive Unfreezing: Khóa backbone 5 epoch đầu
        unfreeze = (epoch >= cfg['train']['warmup_epochs'])
        for param in model.backbone.parameters():
            param.requires_grad = unfreeze

        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Mixup Augmentation
            imgs_m, labels_a, labels_b, lam = mixup_data(imgs, labels)
            
            # Label Smoothing thủ công
            la_s = labels_a * (1 - sm) + 0.5 * sm
            lb_s = labels_b * (1 - sm) + 0.5 * sm
            
            optimizer.zero_grad()
            outputs = model(imgs_m)
            # Tính loss tổng hợp từ Mixup
            loss = lam * criterion(outputs, la_s) + (1 - lam) * criterion(outputs, lb_s)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Tránh nổ gradient
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, all_probs, all_gts = 0.0, [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
                all_probs.extend(torch.sigmoid(out).cpu().numpy())
                all_gts.extend(labels.cpu().numpy())

        # 4. Tính toán Metrics và Giám sát tài nguyên
        m = compute_metrics(np.array(all_gts), np.array(all_probs))
        ram, vram = get_sys_stats()
        scheduler.step()
        
        epoch_time = time.time() - start_time
        log_entry = {
            "epoch": epoch + 1, 
            "train_loss": train_loss / len(train_loader), 
            "val_loss": val_loss / len(val_loader),
            "val_hter": m['hter'], 
            "val_acc": m['acc'], 
            "ram_mb": ram, 
            "vram_mb": vram, 
            "time_sec": epoch_time
        }
        history.append(log_entry)
        save_training_log(history, results_dir)

        print(f"\nEnd Epoch {epoch+1}: T_Loss: {log_entry['train_loss']:.4f} | V_Loss: {log_entry['val_loss']:.4f} | HTER: {m['hter']:.4f} | Acc: {m['acc']:.4f}")
        print(f"Memory: RAM {ram:.0f}MB, VRAM {vram:.0f}MB | Time: {epoch_time:.1f}s")

        # 5. Checkpointing & Early Stopping
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last.pt"))
        if m['hter'] < best_hter:
            best_hter = m['hter']
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
            print(">>> NEW BEST MODEL SAVED")
        else:
            patience_cnt += 1
            print(f">>> Patience Counter: {patience_cnt}/{patience_limit}")

        if patience_cnt >= patience_limit:
            print("Early Stopping Triggered. Kết thúc huấn luyện.")
            break

if __name__ == "__main__":
    main()