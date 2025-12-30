import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random
import sys

# Setup Path
sys.path.append(os.getcwd())
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTLoRA
from src.data.data_loader import build_loaders

# ================= CẤU HÌNH =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results/qualitative_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Đường dẫn file weights của bạn
MODELS_SETUP = {
    "ConvNeXt": {"class": ConvNextBinary, "path": "checkpoints/convnext/best.pt", "size": 224},
    "EfficientNet": {"class": EfficientNetBinary, "path": "checkpoints/efficientnet/best.pt", "size": 260},
    "ViT": {"class": ViTLoRA, "path": "checkpoints/vit/best.pt", "size": 224}
}

# ================= HÀM HỖ TRỢ =================

def load_model_instance(name, cfg):
    model = cfg['class']().to(DEVICE)
    if os.path.exists(cfg['path']):
        state = torch.load(cfg['path'], map_location=DEVICE, weights_only=True)
        # Sửa lỗi prefix module.
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
        model.eval()
        return model
    return None

def get_prediction(model, img_pil, size):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    inp = tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)
        prob = torch.sigmoid(out).item()
    label = 1 if prob >= 0.5 else 0
    return label, prob

def main():
    print(">>> Đang chuẩn bị 10 mẫu ảnh tiêu biểu (5 Live, 5 Spoof)...")
    
    # 1. Chọn mẫu
    test_live_dir = "data/data_split/test/0_live"
    test_spoof_dir = "data/data_split/test/1_spoof"
    
    live_files = [os.path.join(test_live_dir, f) for f in os.listdir(test_live_dir) if f.lower().endswith(('.png', '.jpg'))]
    spoof_files = [os.path.join(test_spoof_dir, f) for f in os.listdir(test_spoof_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    selected_samples = random.sample(live_files, 5) + random.sample(spoof_files, 5)
    
    # 2. Load cả 3 model
    models = {}
    for name, cfg in MODELS_SETUP.items():
        m = load_model_instance(name, cfg)
        if m: models[name] = m

    # 3. Chạy Inference và Vẽ hình
    # Tạo lưới 2 hàng x 5 cột
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    for idx, img_path in enumerate(selected_samples):
        ax = axes[idx // 5, idx % 5]
        img_pil = Image.open(img_path).convert('RGB')
        ax.imshow(img_pil)
        
        gt_label = 0 if "0_live" in img_path else 1
        gt_text = "LIVE" if gt_label == 0 else "SPOOF"
        
        # Tiêu đề ảnh (Ground Truth)
        ax.set_xlabel(f"GT: {gt_text}", fontsize=14, fontweight='bold', labelpad=10)
        ax.set_xticks([]); ax.set_yticks([])
        
        # Dự đoán của 3 model
        results_text = ""
        for name, model in models.items():
            pred_label, prob = get_prediction(model, img_pil, MODELS_SETUP[name]['size'])
            
            # Màu sắc: Xanh nếu đúng, Đỏ nếu sai
            color = "green" if pred_label == gt_label else "red"
            pred_text = "SPOOF" if pred_label == 1 else "LIVE"
            
            # Hiển thị text dự đoán bên dưới mỗi ảnh
            ax.text(0.5, -0.15 - (list(models.keys()).index(name) * 0.1), 
                    f"{name}: {pred_text} ({prob:.2%})", 
                    transform=ax.transAxes, ha="center", fontsize=11, 
                    color=color, fontweight='bold')
        
        # Vẽ khung viền cho ảnh
        rect_color = "black"
        for spine in ax.spines.values():
            spine.set_edgecolor(rect_color)
            spine.set_linewidth(2)

    plt.suptitle("Qualitative Performance Comparison: ConvNeXt vs EfficientNet vs ViT\n(Correct: Green | Incorrect: Red)", 
                 fontsize=22, fontweight='bold', y=0.98)
    
    save_path = os.path.join(OUTPUT_DIR, "Fig8_Qualitative_Comparison_10_samples.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n THÀNH CÔNG! Đã xuất ảnh so sánh đối soát tại: {save_path}")

if __name__ == "__main__":
    main()