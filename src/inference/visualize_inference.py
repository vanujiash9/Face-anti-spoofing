import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Import Model Classes
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTBinary
from src.data.data_loader import build_dataloaders

# ================= CẤU HÌNH FINAL =================
OUTPUT_DIR = "src/results/final_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config cho 3 Model
MODELS_CONFIG = {
    "ConvNeXt": {
        "log_file": "src/results/convnext/training_log.csv",
        "weights": "saved_models/convnext/best.pt",
        "model_class": ConvNextBinary,
        "input_size": 224,
        "color": "blue"
    },
    "EfficientNet": {
        "log_file": "src/results/efficientnet/training_log.csv",
        "weights": "checkpoints/efficientnet/best.pt",
        "model_class": EfficientNetBinary,
        "input_size": 260,
        "color": "green"
    },
    "ViT": {
        "log_file": "src/results/vit/training_log.csv",
        "weights": "checkpoints/vit/best.pt",
        "model_class": ViTBinary,
        "input_size": 224,
        "color": "red"
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_comparison_curves():
    print(">>> DANG VE BIEU DO SO SANH 3 MODEL...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    for name, config in MODELS_CONFIG.items():
        if not os.path.exists(config['log_file']):
            print(f"Skipping {name} (Log not found)")
            continue
            
        df = pd.read_csv(config['log_file'])
        
        # Xử lý tên cột linh hoạt
        cols = df.columns
        val_acc = df['Val_Acc'] if 'Val_Acc' in cols else df['V_Acc']
        val_loss = df['Val_Loss'] if 'Val_Loss' in cols else df['V_Loss']
        epochs = df['Epoch']
        
        # Vẽ Accuracy
        ax1.plot(epochs, val_acc, label=f'{name}', color=config['color'], linewidth=2)
        
        # Vẽ Loss
        ax2.plot(epochs, val_loss, label=f'{name}', color=config['color'], linewidth=2)

    # Trang trí biểu đồ Acc
    ax1.set_title('Validation Accuracy Comparison', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Trang trí biểu đồ Loss
    ax2.set_title('Validation Loss Comparison', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, "all_models_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" -> Da luu bieu do so sanh: {save_path}")

def evaluate_and_heatmap(name, config):
    print(f"\n[{name}] Dang danh gia tren tap Test...")
    
    if not os.path.exists(config['weights']):
        print(f" -> [SKIP] Khong tim thay weights: {config['weights']}")
        return

    try:
        # Load Model
        model = config['model_class'](pretrained=False).to(device)
        state_dict = torch.load(config['weights'], map_location=device)
        
        # Fix DataParallel keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.eval()

        # Load Data
        loaders = build_dataloaders(
            img_dir="data/data_split", 
            input_size=config['input_size'], 
            batch_size=32, 
            num_workers=4
        )
        
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in loaders['test']:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) >= 0.5).long().cpu().numpy()
                y_true.extend(labels.numpy())
                y_pred.extend(preds)

        # Ve Heatmap
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Live', 'Spoof'], yticklabels=['Live', 'Spoof'])
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        
        save_path = os.path.join(OUTPUT_DIR, f"cm_{name}.png")
        plt.savefig(save_path)
        print(f" -> Da luu Heatmap: {save_path}")
        plt.close()
        
        # In Report
        print(classification_report(y_true, y_pred, target_names=['Live', 'Spoof'], digits=4))

    except Exception as e:
        print(f"Loi: {e}")

def main():
    # 1. Vẽ biểu đồ so sánh 3 đường
    plot_comparison_curves()
    
    # 2. Vẽ Heatmap từng cái
    for name, config in MODELS_CONFIG.items():
        evaluate_and_heatmap(name, config)
        
    print(f"\nHOAN TAT! Ket qua luu tai: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()