import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# -------------------------------
# 1. Model
# -------------------------------
class MyModelClass(nn.Module):
    def __init__(self, num_classes=15):
        super(MyModelClass, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

num_classes = 15
model = MyModelClass(num_classes=num_classes)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

class_names = [
    "cay-chuoi-ngoc", "cay-ke-bac", "cay-vu-sua", "hoa-hong", "hoa-hong-mon", "hoa-ram-but"
]

# -------------------------------
# 2. Transform ảnh
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.title("AI Nhận Diện Cây Cảnh 🌿")
st.write("Upload ảnh cây cảnh và nhận dự đoán từ AI")

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh bạn vừa upload", use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    top3_prob, top3_idx = torch.topk(probs, 3)
    st.write("### Top 3 dự đoán:")
    for i in range(3):
        st.write(f"{i+1}. {class_names[top3_idx[i]]} ({top3_prob[i].item()*100:.2f}%)")
