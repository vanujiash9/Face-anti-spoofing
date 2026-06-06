from transformers import pipeline
import os

MODEL_DIR = "saved_models/tf_efficientnetv2_b0"
os.makedirs(MODEL_DIR, exist_ok=True)

model_name = "timm/tf_efficientnetv2_b0.in1k"

from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained(model_name)
model.save_pretrained(MODEL_DIR)

pipe = pipeline(
    "image-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,  
    framework="pt"
)

result = pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")
print(result)
