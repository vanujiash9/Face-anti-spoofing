import os
import psutil
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def get_sys_stats():
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / 1024**2
    vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    return ram, vram

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(float)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    hter = (fpr + (1 - tpr)) / 2
    acc = (tp + tn) / len(y_true)
    return {"acc": acc, "hter": hter, "fpr": fpr, "tpr": tpr, "cm": cm}

def save_training_log(history, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(history)
    log_path = os.path.join(results_dir, "training_log.csv")
    df.to_csv(log_path, index=False)