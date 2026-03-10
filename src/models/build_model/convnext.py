import torch.nn as nn
import timm
from safetensors.torch import load_file

class ConvNextBinary(nn.Module):
    def __init__(self, pretrained_path="saved_models/convnext_tiny_224/model.safetensors"):
        super().__init__()
        # Backbone khong lay head, bat Stochastic Depth (drop_path_rate) de chong overfit
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0, drop_path_rate=0.2)
        state_dict = load_file(pretrained_path)
        self.backbone.load_state_dict(state_dict, strict=False)
        
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)