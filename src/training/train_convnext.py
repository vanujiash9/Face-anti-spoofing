import os, time, yaml, torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.data.data_loader import build_loaders
from src.models.build_model.convnext import ConvNextBinary
from src.training.utils import get_sys_stats, compute_metrics, save_training_log

def main():
    with open("config/convnext.yaml") as f: cfg = yaml.safe_load(f)
    device = torch.device(cfg['train']['device'])
    os.makedirs(cfg['train']['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['train']['results_dir'], exist_ok=True)

    train_loader, val_loader = build_loaders(cfg['dataset']['img_dir'], cfg['dataset']['input_size'], cfg['dataset']['batch_size'], num_workers=cfg['dataset']['num_workers'])
    model = ConvNextBinary().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['train']['learning_rate']), weight_decay=float(cfg['train']['weight_decay']))
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])

    history, best_hter, patience_cnt = [], 1.0, 0
    patience_limit = cfg['train']['patience']
    sm = cfg['train']['label_smoothing']

    print("--- STARTING CONVNEXT TRAINING ---")
    for epoch in range(cfg['train']['epochs']):
        start_time = time.time()
        unfreeze = (epoch >= cfg['train']['warmup_epochs'])
        for param in model.backbone.parameters(): param.requires_grad = unfreeze

        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            labels_s = labels * (1 - sm) + 0.5 * sm
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels_s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        probs, gts, val_loss = [], [], 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
                probs.extend(torch.sigmoid(out).cpu().numpy())
                gts.extend(labels.cpu().numpy()) # FIXED: .cpu() added

        m = compute_metrics(np.array(gts), np.array(probs))
        ram, vram = get_sys_stats()
        scheduler.step()
        log = {"epoch": epoch+1, "train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "hter": m['hter'], "acc": m['acc'], "ram_mb": ram, "vram_mb": vram, "time_sec": time.time()-start_time}
        history.append(log); save_training_log(history, cfg['train']['results_dir'])
        print(f"Ep {epoch+1} | T_Loss: {log['train_loss']:.4f} | V_Loss: {log['val_loss']:.4f} | Acc: {m['acc']:.4f} | HTER: {m['hter']:.4f}")
        
        torch.save(model.state_dict(), os.path.join(cfg['train']['checkpoint_dir'], "last.pt"))
        if m['hter'] < best_hter:
            best_hter = m['hter']
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(cfg['train']['checkpoint_dir'], "best.pt"))
            print(f">>> New Best HTER: {best_hter:.4f} | Model Saved")
        else:
            patience_cnt += 1
            if patience_cnt >= patience_limit:
                print(f"Early Stopping at Epoch {epoch+1}"); break

if __name__ == "__main__":
    main()