import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import sys

# Setup Path
sys.path.append(os.getcwd())
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTLoRA
from src.data.data_loader import build_loaders

# ================= CẤU HÌNH =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results/attack_analysis_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = "data/data_split/test/1_spoof" # Chỉ soi lỗi trên tập Spoof

MODELS_SETUP = {
    "ConvNeXt": {"class": ConvNextBinary, "path": "checkpoints/convnext/best.pt", "size": 224},
    "EfficientNet": {"class": EfficientNetBinary, "path": "checkpoints/efficientnet/best.pt", "size": 260},
    "ViT": {"class": ViTLoRA, "path": "checkpoints/vit/best.pt", "size": 224}
}

SPOOF_CATEGORIES = {
    'Deepfake': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
    'Mask': ['silicon', 'mask', 'latex', '3D_Mask'],
    'Print': ['Poster', 'Photo', 'A4'],
    'Replay': ['Phone', 'PC', 'Pad']
}

def get_attack_type(fname):
    for cat, keywords in SPOOF_CATEGORIES.items():
        if any(kw.lower() in fname.lower() for kw in keywords): return cat
    return "Other"

# ================= HÀM XỬ LÝ =================

def main():
    print(">>> Đang bắt đầu phân tích lỗi đa mô hình...")
    
    # 1. Thu thập dự đoán của cả 3 model
    all_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png'))]
    results_record = [] # Lưu: {'filename', 'type', 'ConvNeXt_res', 'EffNet_res', 'ViT_res'}

    # Khởi tạo model và dict để lưu dự đoán
    predictions_map = {f: {} for f in all_files}

    for name, cfg in MODELS_SETUP.items():
        print(f"-> Đang chạy Inference cho {name}...")
        model = cfg['class']().to(DEVICE)
        state = torch.load(cfg['path'], map_location=DEVICE, weights_only=True)
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
        model.eval()

        tf = transforms.Compose([
            transforms.Resize((cfg['size'], cfg['size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        with torch.no_grad():
            for f in tqdm(all_files, leave=False):
                img_path = os.path.join(DATA_DIR, f)
                img = Image.open(img_path).convert('RGB')
                inp = tf(img).unsqueeze(0).to(DEVICE)
                
                out = torch.sigmoid(model(inp)).item()
                # 1 = Đúng (Bắt được spoof), 0 = Sai (Bị lừa là Real)
                predictions_map[f][name] = 1 if out >= 0.5 else 0
                predictions_map[f][f"{name}_score"] = out

    # 2. Chuyển đổi sang DataFrame để phân tích
    data_list = []
    for f, preds in predictions_map.items():
        row = {'filename': f, 'Attack_Type': get_attack_type(f)}
        row.update(preds)
        data_list.append(row)
    
    df = pd.DataFrame(data_list)
    df.to_csv(f"{OUTPUT_DIR}/detailed_predictions.csv", index=False)

    # 3. TRỰC QUAN HÓA: Độ chính xác theo loại Spoof
    print(">>> Đang vẽ biểu đồ hiệu năng theo loại tấn công...")
    plot_data = []
    for m in MODELS_SETUP.keys():
        for cat in SPOOF_CATEGORIES.keys():
            sub_df = df[df['Attack_Type'] == cat]
            if len(sub_df) > 0:
                acc = sub_df[m].mean() * 100
                plot_data.append({"Model": m, "Category": cat, "Accuracy (%)": acc})
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=pd.DataFrame(plot_data), x="Category", y="Accuracy (%)", hue="Model", palette="muted")
    plt.ylim(85, 101)
    plt.title("Model Robustness by Spoof Attack Category", fontsize=15, fontweight='bold')
    for p in ax.patches: ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='baseline', fontsize=9, xytext=(0, 5), textcoords='offset points')
    plt.savefig(f"{OUTPUT_DIR}/Fig_Attack_Accuracy.png", dpi=300)
    plt.close()

    # 4. TÌM KIẾM CA "SIÊU KHÓ" (Cả 3 mô hình cùng đoán SAI)
    # Lỗi sai ở đây là predict = 0 (vì đây là tập Spoof)
    consensus_fails = df[(df['ConvNeXt'] == 0) & (df['EfficientNet'] == 0) & (df['ViT'] == 0)]
    
    print(f"!!! Tìm thấy {len(consensus_fails)} mẫu cả 3 mô hình đều dự đoán sai (Consensus Failure).")
    
    if not consensus_fails.empty:
        # Vẽ gallery 8 mẫu sai nặng nhất
        num_show = min(len(consensus_fails), 8)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        plt.suptitle("Consensus Failures: Samples misclassified by ALL 3 models", fontsize=20, color='red', fontweight='bold')
        
        for i in range(num_show):
            ax = axes[i//4, i%4]
            f = consensus_fails.iloc[i]['filename']
            img = Image.open(os.path.join(DATA_DIR, f))
            ax.imshow(img)
            # Lấy score trung bình để hiển thị độ "ngu" của AI
            avg_score = (consensus_fails.iloc[i]['ConvNeXt_score'] + consensus_fails.iloc[i]['EfficientNet_score'] + consensus_fails.iloc[i]['ViT_score']) / 3
            ax.set_title(f"Type: {consensus_fails.iloc[i]['Attack_Type']}\nAvg Score: {avg_score:.4f}", color='darkred', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{OUTPUT_DIR}/Fig_Consensus_Failures.png", dpi=300)
        plt.close()

    # 5. LIÊT KÊ CÁC HÌNH SAI RIÊNG BIỆT (Vẽ riêng cho từng model)
    for m in MODELS_SETUP.keys():
        m_fails = df[df[m] == 0].sort_values(by=f"{m}_score").head(4) # 4 mẫu tệ nhất của từng con
        if not m_fails.empty:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            plt.suptitle(f"Top failure cases for {m}", fontsize=16, fontweight='bold')
            for i in range(len(m_fails)):
                f = m_fails.iloc[i]['filename']
                axes[i].imshow(Image.open(os.path.join(DATA_DIR, f)))
                axes[i].set_title(f"Type: {m_fails.iloc[i]['Attack_Type']}\nScore: {m_fails.iloc[i][f'{m}_score']:.4f}")
                axes[i].axis('off')
            plt.savefig(f"{OUTPUT_DIR}/Fig_Failures_{m}.png", dpi=300)
            plt.close()

    print(f"\n HOÀN TẤT! Kết quả đã lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()