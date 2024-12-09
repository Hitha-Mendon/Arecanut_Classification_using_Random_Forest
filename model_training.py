import os
import joblib
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Paths and Parameters
DATASET_DIR = "./dataset"
IMAGE_SIZE = (64, 64)  # Resize all images to 64x64

# Load Dataset
def load_data():
    labels = []
    data = []
    categories = ['high', 'medium', 'low']

    for category in categories:
        path = os.path.join(DATASET_DIR, category)
        class_num = categories.index(category)  # 0: high, 1: medium, 2: low
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)  # Read image using OpenCV
                img_resized = cv2.resize(img, IMAGE_SIZE)  # Resize the image
                data.append(img_resized.flatten())  # Flatten for ML model
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    return np.array(data), np.array(labels)

# Load data
X, y = load_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'arecanut_model.pkl')
print("Model saved as arecanut_model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
