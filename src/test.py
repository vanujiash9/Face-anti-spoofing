import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ================= CONFIG HÀN LÂM =================
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thiết lập font chữ phong cách LaTeX (STIX) và ticks hướng vào trong
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True
})

# Thông tin các mô hình
models_info = [
    {"name": "ConvNeXt", "path": r"scripts\results\convnext\training_log.csv", "color": "#48556e"},
    {"name": "EfficientNet", "path": r"scripts\results\efficientnet\training_log.csv", "color": "#1d7874"},
    {"name": "ViT", "path": r"scripts\results\vit\training_log.csv", "color": "#67b372"}
]

def draw_model_report(info):
    name = info["name"]
    path = info["path"]
    main_color = info["color"]

    if not os.path.exists(path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file {path}")
        return

    # Đọc dữ liệu
    df = pd.read_csv(path)
    
    # --- TỰ ĐỘNG TÌM CỘT ACCURACY ---
    acc_col = None
    for col in df.columns:
        if 'acc' in col.lower(): # Tìm cột nào có chứa chữ 'acc'
            acc_col = col
            break
    
    if acc_col is None:
        print(f"❌ Lỗi: Không tìm thấy cột accuracy trong file {name}. Các cột hiện có: {list(df.columns)}")
        return

    # Tạo khung hình: 1 hàng, 2 cột
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    plt.subplots_adjust(wspace=0.25)

    # --- BIỂU ĐỒ TRÁI: LOSS ---
    ax1.plot(df['epoch'], df['train_loss'], 
             label='Training Loss', color=main_color, 
             linestyle='-', linewidth=2, marker='o', markersize=3, alpha=0.8)
    ax1.plot(df['epoch'], df['val_loss'], 
             label='Validation Loss', color='#d62728', 
             linestyle='--', linewidth=2, marker='x', markersize=4, alpha=0.9)
    
    ax1.set_title(f'{name}: Loss Convergence', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.set_ylim(0.0, 0.8)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.legend(frameon=True)
    ax1.grid(False)

    # --- BIỂU ĐỒ PHẢI: ACCURACY ---
    ax2.plot(df['epoch'], df[acc_col], 
             label='Validation Accuracy', color=main_color, 
             linestyle='-', linewidth=2, marker='s', markersize=4)
    
    ax2.set_title(f'{name}: Accuracy Progression', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy Score')
    ax2.set_ylim(0.5, 1.02) 
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(False)

    # Lưu ảnh riêng
    save_filename = f'Training_Report_{name}.png'
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã xuất báo cáo thành công cho {name} tại: {save_path}")

# Thực hiện vẽ cho 3 model
for info in models_info:
    draw_model_report(info)