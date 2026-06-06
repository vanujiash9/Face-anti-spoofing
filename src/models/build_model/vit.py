import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

class ViTLoRA(nn.Module):
    def __init__(self, pretrained_path="saved_models/vit_base_patch16_224/model.safetensors"):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        state_dict = load_file(pretrained_path)
        self.backbone.load_state_dict(state_dict, strict=False)
        
        # LoRA fine-tuning cho Transformer
        config = LoraConfig(r=16, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.1)
        self.backbone = get_peft_model(self.backbone, config)
        
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)