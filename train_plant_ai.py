import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Thư mục dữ liệu
data_dir = "plants_dataset"

# Chuẩn bị dữ liệu
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=16,
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=16,
    subset='validation'
)

# Mô hình đơn giản
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện
model.fit(train_data, validation_data=val_data, epochs=10)

# Lưu mô hình
model.save("plant_model.h5")

print("✅ Huấn luyện xong! Mô hình đã lưu thành plant_model.h5")
