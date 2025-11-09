# train_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# --- Cấu hình ---
MODEL_PATH = "model.pth"
NUM_CLASSES = 5  # số lớp cây, ví dụ: Luoi_ho, Kim_ngan, Trau_ba, Sen_da, Xuong_rong

# --- Khởi tạo model với weights mới, không cảnh báo deprecated ---
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

# Thay lớp cuối cho số lớp cây của bạn
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# --- Lưu model ngay, không cần train (demo) ---
torch.save(model, MODEL_PATH)
print(f"✅ Model demo đã được lưu thành: {MODEL_PATH}")
