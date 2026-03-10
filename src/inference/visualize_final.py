import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

# ================= CẤU HÌNH =================
BASE_DIR = "results/final_comparison"
OUTPUT_DIR = "results/final_visual_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dữ liệu bạn vừa chạy (Cập nhật chính xác từ Log của bạn)
data = {
    'Model': ['ConvNeXt', 'ConvNeXt', 'ConvNeXt', 'ConvNeXt', 
              'EfficientNet', 'EfficientNet', 'EfficientNet', 'EfficientNet',
              'ViT', 'ViT', 'ViT', 'ViT'],
    'Attack Type': ['Deepfake', 'Mask', 'Print', 'Replay',
                   'Deepfake', 'Mask', 'Print', 'Replay',
                   'Deepfake', 'Mask', 'Print', 'Replay'],
    'Accuracy': [100.0, 100.0, 98.83, 99.78, 
                 100.0, 99.73, 98.83, 99.35, 
                 100.0, 99.46, 97.47, 98.70]
}

def plot_attack_comparison():
    print(">>> Đang vẽ biểu đồ so sánh các nhóm tấn công...")
    df = pd.DataFrame(data)
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Vẽ biểu đồ cột
    ax = sns.barplot(data=df, x='Attack Type', y='Accuracy', hue='Model', palette='viridis')
    
    # Chỉnh dải Y từ 95% để thấy rõ sự chênh lệch
    plt.ylim(95, 100.5)
    plt.title("So sánh độ chính xác theo từng loại tấn công", fontsize=15, fontweight='bold')
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='lower right')
    
    # Ghi số lên đầu cột
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_attack_type_comparison.png"), dpi=300)
    plt.close()

def plot_error_heatmap():
    print(">>> Đang vẽ Heatmap tỷ lệ lỗi...")
    # Chuyển Accuracy thành Error Rate (100 - Acc)
    df = pd.DataFrame(data)
    df['Error Rate'] = 100 - df['Accuracy']
    
    pivot_df = df.pivot(index="Attack Type", columns="Model", values="Error Rate")
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap="YlOrRd", fmt=".2f", linewidths=0.5)
    plt.title("Bản đồ tỷ lệ lỗi (%) - Màu càng đỏ lỗi càng cao", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_error_heatmap.png"), dpi=300)
    plt.close()

def show_worst_failures():
    print(">>> Đang tìm và hiển thị các mẫu ảnh dự đoán sai...")
    # Quét các file CSV failed đã tạo từ bước trước
    models = ["ConvNeXt", "EfficientNet", "ViT"]
    all_failed = []

    for m in models:
        csv_path = os.path.join(BASE_DIR, f"{m}_failed_spoof.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['model_failed'] = m
            all_failed.append(df)

    if not all_failed:
        print("Không tìm thấy file lỗi CSV.")
        return

    full_failed_df = pd.concat(all_failed)
    
    # Tìm những ảnh mà nhiều model cùng đoán sai (Hard samples)
    # Group theo path và đếm số model đoán sai
    common_fails = full_failed_df.groupby('path').size().reset_index(name='fail_count')
    common_fails = common_fails.sort_values(by='fail_count', ascending=False).head(8)

    if common_fails.empty:
        print("Tuyệt vời, không có ảnh nào bị sai chung!")
        return

    # Vẽ lưới ảnh lỗi
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    plt.suptitle("Top các mẫu ảnh giả bị lọt lưới (Mô hình nhận diện sai)", fontsize=18, fontweight='bold', color='red')
    
    for i, (idx, row) in enumerate(common_fails.iterrows()):
        ax = axes[i//4, i%4]
        try:
            img_path = row['path']
            img = Image.open(img_path)
            ax.imshow(img)
            
            # Lấy thông tin loại spoof từ path
            cat = img_path.split('_')[-1].split('.')[0]
            ax.set_title(f"Loại: {cat}\nSai bởi {row['fail_count']} models", color='darkred', fontsize=12)
        except:
            ax.text(0.5, 0.5, "Image not found")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_failure_gallery.png"), dpi=300)
    plt.close()
    print(f"Đã lưu thư viện ảnh lỗi tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_attack_comparison()
    plot_error_heatmap()
    show_worst_failures()
    print(f"\nHOÀN TẤT! Toàn bộ báo cáo trực quan nằm tại: {OUTPUT_DIR}")