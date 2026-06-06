import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from collections import defaultdict

# ================= CẤU HÌNH =================
DATA_DIR = os.path.join('data', 'data_process', 'cropped_faces')
OUTPUT_DIR = os.path.join('data', 'data_process')

# Định nghĩa các nhóm tấn công để gộp nhóm (Mapping)
SPOOF_CATEGORIES = {
    'Deepfake': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter', 'CelebA'],
    'Mask': ['silicon', 'mask', 'latex', 'UpperBodyMask', 'RegionMask', '3D_Mask'],
    'Print': ['Poster', 'Photo', 'A4', 'Print'],
    'Replay': ['Phone', 'PC', 'Pad', 'Screen', 'Replay']
}

def get_spoof_category(filename):
    """Phân loại chi tiết dựa trên tên file"""
    # Filename format: Dataset_ID_Label_Type.png
    name_body = os.path.splitext(filename)[0]
    parts = name_body.split('_')
    
    if len(parts) < 2: return 'Unknown'
    
    # Label check
    is_live = False
    if parts[-2] == '0': is_live = True
    
    if is_live: return 'Live'
    
    # Spoof Type check
    s_type = parts[-1] # Lấy phần cuối cùng
    
    # Map s_type vào nhóm lớn
    for category, keywords in SPOOF_CATEGORIES.items():
        for kw in keywords:
            if kw.lower() in s_type.lower():
                return category
    
    # Nếu không thuộc nhóm nào ở trên, trả về chính tên loại đó
    return s_type

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found")
        return

    print("Dang quet va phan loai du lieu...")
    
    # Dictionary lưu đường dẫn ảnh theo nhóm
    data_groups = defaultdict(list)
    sizes = [] # Lưu kích thước ảnh để vẽ biểu đồ
    
    # Duyệt file
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(('jpg', 'png', 'jpeg')):
                full_path = os.path.join(root, f)
                category = get_spoof_category(f)
                data_groups[category].append(full_path)
                
                # Lấy sample size (để tránh đọc hết ổ cứng gây chậm)
                if random.random() < 0.1: # Lấy mẫu 10% để thống kê size
                    try:
                        with Image.open(full_path) as img:
                            sizes.append(img.size[0]) # Lấy width
                    except: pass

    # ================= HÌNH 1: MA TRẬN CÁC LOẠI TẤN CÔNG =================
    print("Dang ve Hinh 1: Cac loai tan cong...")
    
    # Chọn các nhóm tiêu biểu để hiển thị
    target_groups = ['Live', 'Deepfake', 'Mask', 'Replay', 'Print']
    cols = 5 # Số ảnh mỗi hàng
    
    fig, axes = plt.subplots(nrows=len(target_groups), ncols=cols, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for row_idx, group_name in enumerate(target_groups):
        images = data_groups.get(group_name, [])
        # Nếu nhóm rỗng hoặc ít ảnh (do tên file không khớp), lấy ngẫu nhiên từ nhóm 'Spoof' chung
        if len(images) < cols:
            # Fallback logic nếu cần
            pass
            
        sample_imgs = random.sample(images, min(len(images), cols))
        
        # Tiêu đề hàng (Bên trái)
        axes[row_idx, 0].set_ylabel(group_name, fontsize=16, fontweight='bold', labelpad=20)
        
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col_idx < len(sample_imgs):
                img_path = sample_imgs[col_idx]
                try:
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    
                    # Lấy tên cụ thể để chú thích nhỏ bên dưới
                    specific_type = os.path.splitext(os.path.basename(img_path))[0].split('_')[-1]
                    if group_name == 'Live': specific_type = "Real"
                    ax.set_xlabel(specific_type, fontsize=9)
                except:
                    pass
            else:
                ax.axis('off') # Ẩn nếu không có ảnh

    fig.suptitle('Visualization of Attack Types in Dataset', fontsize=20, y=0.95)
    save_path1 = os.path.join(OUTPUT_DIR, 'paper_fig_1_categories.png')
    plt.savefig(save_path1, bbox_inches='tight', dpi=300) # DPI 300 chuẩn in ấn
    print(f" -> Da luu: {save_path1}")


    # ================= HÌNH 2: PHÂN BỐ KÍCH THƯỚC ẢNH =================
    print("Dang ve Hinh 2: Phan bo kich thuoc...")
    plt.figure(figsize=(10, 6))
    
    # Vẽ histogram
    plt.hist(sizes, bins=30, color='#4CAF50', edgecolor='black', alpha=0.7)
    plt.axvline(x=224, color='red', linestyle='dashed', linewidth=2, label='Min Input (224px)')
    
    plt.title('Image Resolution Distribution (Width)', fontsize=14)
    plt.xlabel('Pixel Width', fontsize=12)
    plt.ylabel('Number of Images (Sampled)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    save_path2 = os.path.join(OUTPUT_DIR, 'paper_fig_2_distribution.png')
    plt.savefig(save_path2, dpi=300)
    print(f" -> Da luu: {save_path2}")


    # ================= HÌNH 3: PIXEL INTENSITY (LIVE vs SPOOF) =================
    # Hình này chứng minh sự khác biệt về phổ màu (Paper hay dùng)
    print("Dang ve Hinh 3: Pixel Histogram...")
    
    def get_avg_hist(img_paths):
        avg_hist = np.zeros(256)
        count = 0
        for p in img_paths:
            try:
                img = Image.open(p).convert('L') # Chuyển xám
                hist = img.histogram()
                avg_hist += np.array(hist)
                count += 1
            except: pass
        return avg_hist / count if count > 0 else avg_hist

    # Lấy mẫu để tính toán cho nhanh
    live_samples = random.sample(data_groups['Live'], min(len(data_groups['Live']), 200))
    # Gom tất cả các loại spoof lại
    all_spoofs = []
    for k, v in data_groups.items():
        if k != 'Live': all_spoofs.extend(v)
    spoof_samples = random.sample(all_spoofs, min(len(all_spoofs), 200))

    hist_live = get_avg_hist(live_samples)
    hist_spoof = get_avg_hist(spoof_samples)

    plt.figure(figsize=(10, 6))
    plt.plot(hist_live, color='green', label='Live Faces', linewidth=2)
    plt.plot(hist_spoof, color='red', label='Spoof Faces', linewidth=2, linestyle='--')
    plt.fill_between(range(256), hist_live, color='green', alpha=0.1)
    
    plt.title('Pixel Intensity Distribution (Grayscale)', fontsize=14)
    plt.xlabel('Pixel Value (0-255)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    
    save_path3 = os.path.join(OUTPUT_DIR, 'paper_fig_3_pixel_hist.png')
    plt.savefig(save_path3, dpi=300)
    print(f" -> Da luu: {save_path3}")
    
    # Show
    plt.show()

if __name__ == "__main__":
    main()