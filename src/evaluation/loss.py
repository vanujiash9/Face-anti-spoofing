import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ================= CONFIG HÀN LÂM (LaTeX Style) =================
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

# Bảng màu đã thống nhất: ConvNeXt (Blue), EfficientNet (Teal), ViT (Green)
colors = {"ConvNeXt": "#48556e", "EfficientNet": "#1d7874", "ViT": "#67b372"}

# ================= DỮ LIỆU TỪ LOG CỦA BẠN =================
# (Dữ liệu này được trích xuất từ nội dung bạn gửi)

vit_data = {
    'epoch': list(range(1, 31)),
    'train': [0.6573, 0.6538, 0.6513, 0.6243, 0.5377, 0.4825, 0.4377, 0.4057, 0.3876, 0.3700, 0.3589, 0.3473, 0.3352, 0.3263, 0.3242, 0.3156, 0.3089, 0.3048, 0.2989, 0.2939, 0.2892, 0.2878, 0.2812, 0.2777, 0.2754, 0.2718, 0.2729, 0.2719, 0.2692, 0.2719],
    'val': [0.6495, 0.6352, 0.6345, 0.5218, 0.4168, 0.3646, 0.2926, 0.2772, 0.3019, 0.2301, 0.2622, 0.2071, 0.2357, 0.2605, 0.2461, 0.2020, 0.2219, 0.1669, 0.1683, 0.1836, 0.1826, 0.1575, 0.1626, 0.1642, 0.1556, 0.1595, 0.1686, 0.1578, 0.1619, 0.1599]
}

convnext_data = {
    'epoch': list(range(1, 18)),
    'train': [0.5026, 0.4783, 0.4735, 0.4677, 0.4622, 0.4547, 0.4400, 0.4315, 0.4327, 0.4257, 0.4250, 0.4179, 0.4183, 0.4132, 0.4150, 0.4117, 0.4125],
    'val': [0.3115, 0.2958, 0.2901, 0.2768, 0.2659, 0.2267, 0.2170, 0.1962, 0.2013, 0.1834, 0.1896, 0.1787, 0.1854, 0.1703, 0.1688, 0.1635, 0.1713]
}

eff_data = {
    'epoch': list(range(1, 23)),
    'train': [0.6195, 0.5798, 0.5664, 0.5453, 0.5299, 0.5129, 0.5031, 0.4878, 0.4745, 0.4636, 0.4557, 0.4436, 0.4405, 0.4325, 0.4229, 0.4187, 0.4130, 0.4077, 0.4078, 0.3994, 0.3970, 0.3926],
    'val': [0.5378, 0.5108, 0.4819, 0.4668, 0.4526, 0.4256, 0.4026, 0.3864, 0.3772, 0.3647, 0.3627, 0.3419, 0.3692, 0.3322, 0.3294, 0.3328, 0.3105, 0.3126, 0.3131, 0.2925, 0.2887, 0.2945]
}

all_logs = [
    ("ConvNeXt", convnext_data),
    ("EfficientNet", eff_data),
    ("ViT", vit_data)
]

# ================= VẼ BIỂU ĐỒ =================
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
plt.subplots_adjust(wspace=0.25)

for i, (name, data) in enumerate(all_logs):
    ax = axes[i]
    color = colors[name]
    
    # Vẽ Train Loss (Đường liền)
    ax.plot(data['epoch'], data['train'], label='Train Loss', 
            color=color, linestyle='-', linewidth=2, marker='o', markersize=4, alpha=0.8)
    
    # Vẽ Val Loss (Đường đứt nét)
    ax.plot(data['epoch'], data['val'], label='Val Loss', 
            color='#d62728', linestyle='--', linewidth=2, marker='x', markersize=4, alpha=0.9)
    
    ax.set_title(f'Training Dynamics: {name}', fontweight='bold', pad=15)
    ax.set_xlabel('Epochs')
    if i == 0:
        ax.set_ylabel('Loss Value')
    
    ax.legend(frameon=True, loc='upper right')
    ax.grid(False) # Bỏ grid theo yêu cầu
    
    # Giới hạn trục Y để biểu đồ trông cân đối
    ax.set_ylim(0, 0.8)
    
    # Tinh chỉnh ticks hướng vào trong
    ax.tick_params(direction='in', length=5)

plt.suptitle('Learning Convergence Comparison across Architectures', fontsize=16, y=1.02, fontweight='bold')
plt.savefig(f'{OUTPUT_DIR}/Loss_Curves_Comparison.png', dpi=300, bbox_inches='tight')

print(f" Đã xuất biểu đồ Loss Curves tại: {OUTPUT_DIR}/Loss_Curves_Comparison.png")