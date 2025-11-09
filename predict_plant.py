import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load mô hình
model = tf.keras.models.load_model("plant_model.h5")

# Danh sách các loại cây theo đúng thứ tự thư mục con trong plants_dataset
class_names = ['cay-chuoi-ngoc', 'cay-ke-bac', 'cay-tung-thom', 'cây vú sữa', 'hoa-hong-mon', 'hoa-ram-but']  # chỉnh lại theo dataset thật của bạn

# Đường dẫn ảnh cần nhận dạng
img_path = "test.jpg"

# Chuẩn bị ảnh
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Dự đoán
pred = model.predict(img_array)
predicted_class = class_names[np.argmax(pred)]

print("🌿 Ảnh này là:", predicted_class)
