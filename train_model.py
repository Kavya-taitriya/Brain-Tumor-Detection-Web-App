import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load and preprocess dataset
data = []
labels = []
img_size = 150

dataset_path = "brain_tumor_dataset"

for category in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, category)
    label = 1 if category == "yes" else 0

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label)
        except Exception as e:
            print("Error loading image:", img_path)

# Convert to numpy and normalize
data = np.array(data) / 255.0
labels = np.array(labels)

# Split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Save model
model.save("brain_tumor_model.h5")
print("✅ Model trained and saved as brain_tumor_model.h5")
