import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ================= CONFIG HÀN LÂM (LaTeX Style) =================
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thiết lập Font mô phỏng LaTeX (Computer Modern / STIX)
plt.rcParams.update({
    "text.usetex": False, # Đặt True nếu máy bạn đã cài LaTeX (MiKTeX/TeXLive)
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# Màu sắc chuyên nghiệp (Deep Blue, Teal, Green)
colors = ["#48556e", "#1d7874", "#67b372"] 

# DATA
models = ['ConvNeXt', 'EfficientNet', 'ViT']
metrics_data = {
    'Accuracy': [0.9520, 0.9892, 0.8889],
    'Precision': [0.9752, 0.9986, 0.9281],
    'Recall': [0.9436, 0.9834, 0.8822],
    'F1-score': [0.9591, 0.9909, 0.9045]
}
attack_data = {
    'Print': [0.9398, 0.9883, 0.8796],
    'Replay': [0.9050, 0.9395, 0.8683],
    'Mask': [0.9537, 0.9974, 0.9061],
    'Deepfake': [0.9720, 1.0000, 0.8575]
}
cms = [
    np.array([[1411, 52], [122, 2042]]), # ConvNeXt
    np.array([[1460, 3], [36, 2128]]),   # EfficientNet
    np.array([[1315, 148], [255, 1909]]) # ViT
]

def add_labels(ax):
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        xytext=(0, 5), textcoords='offset points')

# --- 1. BIỂU ĐỒ CHỈ SỐ NHỊ PHÂN ---
df_m = pd.DataFrame(metrics_data, index=models).reset_index().melt(id_vars='index')
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_m, x='variable', y='value', hue='index', palette=colors, ax=ax1, edgecolor="black", linewidth=0.5)
add_labels(ax1)
ax1.set_title('Comparative Analysis of Binary Classification Performance Metrics', pad=20)
ax1.set_ylim(0.8, 1.0)
ax1.set_ylabel('Score Value')
ax1.set_xlabel('')
ax1.legend(title='Architectures', loc='lower right', frameon=True)
ax1.grid(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Metrics_Comparison_LaTeX.png', dpi=300)

# --- 2. BIỂU ĐỒ TỈ LỆ THEO ATTACK VECTORS ---
df_a = pd.DataFrame(attack_data, index=models).reset_index().melt(id_vars='index')
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_a, x='variable', y='value', hue='index', palette=colors, ax=ax2, edgecolor="black", linewidth=0.5)
add_labels(ax2)
ax2.set_title('Model Robustness Evaluation against Specific Attack Vectors', pad=20)
ax2.set_ylim(0.8, 1.0)
ax2.set_ylabel('Recall Rate (Accuracy per Class)')
ax2.set_xlabel('Spoofing Categories')
ax2.legend(title='Architectures', loc='lower right', frameon=True)
ax2.grid(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Attack_Analysis_LaTeX.png', dpi=300)

# --- 3. MA TRẬN NHẦM LẪN (LAYOUT NGANG CHUẨN PAPER) ---
fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.4)

for i, (cm, model) in enumerate(zip(cms, models)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False, square=True,
                xticklabels=['Live (Pred)', 'Spoof (Pred)'], 
                yticklabels=['Live (True)', 'Spoof (True)'],
                annot_kws={"size": 13, "fontweight": "bold"})
    
    axes[i].set_title(f'Confusion Matrix: {model}', fontsize=12, pad=10)
    axes[i].tick_params(axis='both', which='both', length=0) # Bỏ gạch nhỏ ở trục
    # Vẽ khung cho ma trận
    for _, spine in axes[i].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

plt.suptitle('Detailed Classification Results via Confusion Matrices', fontsize=16, y=0.95)
plt.savefig(f'{OUTPUT_DIR}/Confusion_Matrices_LaTeX.png', dpi=300, bbox_inches='tight')

print(f" Đã xuất bản bộ biểu đồ phong cách LaTeX tại thư mục: {OUTPUT_DIR}")