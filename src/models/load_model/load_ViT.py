from transformers import AutoFeatureExtractor, AutoModelForImageClassification, pipeline
import torch
import os

MODEL_DIR =  "saved_models/vit_base_patch16_224"
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

feature_extractor.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)

pipe = pipeline(
    "image-classification",
    model=MODEL_DIR,
    feature_extractor=MODEL_DIR,
    framework="pt",  
    device=0 if torch.cuda.is_available() else -1
)

# Test với một ảnh
result = pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")
print(result)
