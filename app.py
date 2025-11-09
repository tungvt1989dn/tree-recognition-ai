# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

# --- Load model ---
model = torch.load('model.pth', map_location='cpu')
model.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Danh sách cây ---
class_names = ["cay-chuoi-ngoc", "cay-ke-bac", "cay-vu-sua", "hoa-hong", "hoa-hong-mon", "hoa-ram-but"]

# --- Giao diện Streamlit ---
st.set_page_config(page_title="AI Nhận diện cây cảnh 🌿", layout="wide")
st.title("🌿 AI Nhận diện cây cảnh")

uploaded = st.file_uploader("Tải ảnh cây cảnh lên:", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Ảnh bạn đã tải lên", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_class = probs.max(1)

    st.success(f"✅ Dự đoán: {class_names[top_class.item()]} ({top_prob.item()*100:.2f}%)")
