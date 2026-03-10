import torch.nn as nn
import timm
from safetensors.torch import load_file

class EfficientNetBinary(nn.Module):
    def __init__(self, pretrained_path="saved_models/tf_efficientnetv2_b0/model.safetensors"):
        super().__init__()
        # Khởi tạo backbone không lấy head (num_classes=0 trả về vector 1280 chiều)
        self.backbone = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=0)
        
        # Nạp trọng số cục bộ
        state_dict = load_file(pretrained_path)
        self.backbone.load_state_dict(state_dict, strict=False)
        
        self.num_features = self.backbone.num_features # 1280 cho bản b0
        
        # Head MLP 2 lớp chuyên biệt chống Overfit
        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)