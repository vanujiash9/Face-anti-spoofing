import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.models.build_model.convnext import ConvNextBinary
from src.data.data_loader import build_loaders

# ================= CẤU HÌNH =================
OUTPUT_DIR = "results/final_paper_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. T-SNE VISUALIZATION (Chỉ làm cho ConvNeXt - Model tốt nhất)
def plot_tsne():
    print(">>> 1. Đang vẽ t-SNE (Trực quan hóa đặc trưng)...")
    model = ConvNextBinary().to(device)
    state = torch.load("checkpoints/convnext/best.pt", map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()

    # Chỉ lấy Backbone để lấy Features (768 chiều)
    _, loader = build_loaders("data/data_split", 224, 32)
    
    features, labels = [], []
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(tqdm(loader, desc="Extracting Features")):
            feat = model.backbone(imgs.to(device))
            if len(feat.shape) == 4: feat = feat.mean(dim=[2,3])
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
            if i > 50: break # Lấy khoảng 1500 mẫu cho đẹp

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
    embeds = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeds[:,0], y=embeds[:,1], hue=labels, palette={0: 'green', 1: 'red'}, alpha=0.6)
    plt.title("t-SNE Visualization of Feature Space (ConvNeXt)")
    plt.savefig(f"{OUTPUT_DIR}/Figure_TSNE.png", dpi=300)
    print(f" -> Đã lưu t-SNE")

# 2. STRESS TEST (Nhiễu & Mờ)
def run_stress_test():
    print(">>> 2. Đang thực hiện Stress Test (Nhiễu & Mờ)...")
    # Chúng ta áp dụng nhiễu trực tiếp vào Transform
    blur_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.GaussianBlur(5, sigma=(1.0, 2.0)), # Làm mờ nặng
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load lại data với transform làm mờ
    from torchvision.datasets import ImageFolder
    test_set = ImageFolder("data/data_split/test", transform=blur_tf)
    loader = DataLoader(test_set, batch_size=32)
    
    model = ConvNextBinary().to(device)
    model.load_state_dict(torch.load("checkpoints/convnext/best.pt", map_location=device))
    model.eval()

    correct = 0
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Testing Blur Robustness"):
            out = model(imgs.to(device))
            preds = (torch.sigmoid(out) >= 0.5).cpu().long()
            correct += (preds == lbls).sum().item()
    
    acc = correct / len(test_set)
    print(f" -> Accuracy khi ảnh bị MỜ: {acc:.4f}")
    with open(f"{OUTPUT_DIR}/stress_test_results.txt", "w") as f:
        f.write(f"Blur Accuracy: {acc:.4f}")

def main():
    plot_tsne()
    run_stress_test()
    print(f"\n HOÀN TẤT TOÀN BỘ CÔNG VIỆC! Bạn có thể tải folder {OUTPUT_DIR} về máy.")

if __name__ == "__main__":
    main()