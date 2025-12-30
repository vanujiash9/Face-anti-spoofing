import os
from transformers import AutoModel, AutoConfig

def load_and_save_convnext():
    model_name = "facebook/convnext-tiny-224"
    save_dir = "saved_models/convnext_tiny_224"

    # Tạo thư mục nếu chưa có
    os.makedirs(save_dir, exist_ok=True)

    print(" Đang tải cấu hình model...")
    config = AutoConfig.from_pretrained(model_name)

    print(" Đang tải model ConvNeXt Tiny 224 từ HuggingFace...")
    model = AutoModel.from_pretrained(model_name)

    print(f" Đang lưu model vào: {save_dir} ...")
    model.save_pretrained(save_dir)

    print("Hoàn thành! Model đã được lưu vào saved_models/convnext_tiny_224")
    print(" Các file gồm: config.json, model.safetensors")

if __name__ == "__main__":
    load_and_save_convnext()
