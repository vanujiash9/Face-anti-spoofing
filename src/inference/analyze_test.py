import os
import sys
import yaml
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import các model
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.vit import ViTLoRA
from src.data.data_loader import build_dataloaders

def analyze_model(config_path):
    # 1. Setup
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    if not os.path.exists(config_path):
        print(f"Lỗi: Không tìm thấy config {config_path}")
        return

    with open(config_path) as f: config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = config['model_name']
    print(f"--- Đang phân tích Model: {model_name.upper()} ---")

    # 2. Load Model & Best Checkpoint
    # Lưu ý: Lúc này ta load checkpoint đã train, không cần load weight gốc nữa
    if model_name == 'efficientnet':
        model = EfficientNetBinary(pretrained_path=None)
    elif model_name == 'convnext':
        model = ConvNextBinary(pretrained_path=None)
    elif model_name == 'vit':
        model = ViTLoRA(pretrained_path=None, lora_rank=config['vit_config']['lora_rank'])
    
    ckpt_path = os.path.join(project_root, config['train']['checkpoint_dir'], "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"Chưa có checkpoint best.pt tại {ckpt_path}, thử lấy last.pt")
        ckpt_path = os.path.join(project_root, config['train']['checkpoint_dir'], "last.pt")
    
    print(f"-> Loading weights: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Load Test Data (Shuffle = False để khớp thứ tự file CSV gốc)
    loaders = build_dataloaders(**config['dataset'])
    test_loader = loaders['test']
    
    # Đọc lại file CSV gốc để lấy tên file ảnh (vì DataLoader chỉ trả về ảnh pixel)
    test_csv_path = os.path.join(project_root, config['dataset']['split_dir'], "test.csv")
    df_test = pd.read_csv(test_csv_path)
    
    # 4. Chạy Inference & Tính Loss từng sample
    results = []
    criterion_none = nn.CrossEntropyLoss(reduction='none') # Tính loss từng cái
    
    print("-> Đang chạy kiểm tra từng ảnh...")
    with torch.no_grad():
        # Dùng zip để duyệt qua cả DataLoader và DataFrame cùng lúc (đảm bảo thứ tự)
        # Lưu ý: Cách này đúng nếu DataLoader không shuffle và không drop_last
        idx_tracker = 0
        
        for imgs, labels in tqdm(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward
            outputs = model(imgs)
            
            # Tính xác suất
            probs = torch.softmax(outputs, dim=1)
            
            # Tính Loss riêng cho từng ảnh
            losses = criterion_none(outputs, labels)
            
            # Lấy dự đoán
            preds = torch.argmax(probs, dim=1)
            
            # Lưu kết quả
            imgs_cpu = imgs.cpu()
            for i in range(len(labels)):
                # Lấy thông tin từ DataFrame gốc dựa vào index
                file_name = df_test.iloc[idx_tracker]['filepath']
                
                gt = labels[i].item()
                pred = preds[i].item()
                prob_live = probs[i][0].item()
                prob_spoof = probs[i][1].item()
                loss_val = losses[i].item()
                
                results.append({
                    "Filename": file_name,
                    "GroundTruth": gt,       # 0: Live, 1: Spoof
                    "Prediction": pred,
                    "Confidence_Live": prob_live,
                    "Confidence_Spoof": prob_spoof,
                    "Loss": loss_val,        # Loss cao -> Model đoán sai hoặc không chắc chắn
                    "Result": "CORRECT" if gt == pred else "WRONG"
                })
                idx_tracker += 1

    # 5. Lưu ra Excel/CSV
    out_dir = os.path.join(project_root, config['train']['results_dir'])
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test_detailed_analysis.csv")
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_file, index=False)
    
    print(f"\n✅ Đã lưu file kết quả chi tiết tại: {out_file}")
    print("-" * 30)
    print("THỐNG KÊ NHANH:")
    print(f"Tổng số ảnh: {len(df_res)}")
    print(f"Số ảnh đoán SAI: {len(df_res[df_res['Result']=='WRONG'])}")
    print("Top 5 ảnh sai có Loss cao nhất (Model tự tin sai):")
    print(df_res[df_res['Result']=='WRONG'].nlargest(5, 'Loss')[['Filename', 'GroundTruth', 'Prediction', 'Loss']])

if __name__ == "__main__":
    # Chọn config model bạn muốn soi
    # analyze_model("config/efficientnet.yaml")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/efficientnet.yaml", help="Path to config")
    args = parser.parse_args()
    
    analyze_model(args.config)