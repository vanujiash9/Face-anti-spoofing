import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from src.data.data_loader_robust import build_loaders
from src.models.build_model import FASModel
from src.training.utils import compute_fas_metrics, get_sys_stats

def run_full_report(model_name, config_path):
    import yaml
    with open(config_path) as f: cfg = yaml.safe_load(f)
    device = torch.device("cuda")
    
    # 1. Load Data & Model
    _, val_loader = build_loaders(cfg['dataset']['img_dir'], cfg['dataset']['input_size'], cfg['dataset']['batch_size'])
    model = FASModel(model_name, cfg).to(device)
    model.load_state_dict(torch.load(cfg['train']['checkpoint']))
    model.eval()

    # 2. Inference & Time Benchmarking
    probs, labels = [], []
    start_time = time.time()
    with torch.no_grad():
        for imgs, lbls in val_loader:
            out = torch.sigmoid(model(imgs.to(device)))
            probs.extend(out.cpu().numpy())
            labels.extend(lbls.numpy())
    
    latency = (time.time() - start_time) / len(labels) * 1000 # ms per image
    mem, _ = get_sys_stats()

    probs, labels = np.array(probs), np.array(labels)

    # 3. Multi-threshold Analysis (0 -> 1)
    thresholds = [round(i * 0.1, 1) for i in range(11)]
    report_data = []
    for t in thresholds:
        m = compute_fas_metrics(labels, probs, threshold=t)
        report_data.append({"Thresh": t, "TPR": m['tpr'], "FPR": m['fpr'], "HTER": m['hter'], "F1": m['f1']})
    
    # 4. TRỰC QUAN HÓA
    # a. Probability Histogram (Do tach biet cua model)
    plt.figure(figsize=(10, 6))
    sns.histplot(probs[labels==0], color="green", kde=True, label="Live")
    sns.histplot(probs[labels==1], color="red", kde=True, label="Spoof")
    plt.title(f"Separability Analysis - {model_name}")
    plt.savefig(f"results/{model_name}_histogram.png")

    # b. Confusion Matrix Heatmap (tai nguong 0.5)
    best_m = compute_fas_metrics(labels, probs, threshold=0.5)
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_m['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=['Live', 'Spoof'], yticklabels=['Live', 'Spoof'])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"results/{model_name}_cm.png")

    # c. Accuracy Gallery (Anh dung/sai)
    # Lay ngau nhien anh de show... (Code nay ban da co phan visualize)

    return pd.DataFrame(report_data), latency, mem

def main():
    # Chay cho ca 3 model va in ra bang so sanh cuoi cung
    for m in ['convnext', 'efficientnet', 'vit']:
        df, lat, mem = run_full_report(m, f"config/{m}.yaml")
        print(f"\n--- MODEL: {m.upper()} ---")
        print(f"Latency: {lat:.2f}ms/img | Memory: {mem:.0f}MB")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()