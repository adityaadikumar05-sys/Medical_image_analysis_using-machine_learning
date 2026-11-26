import os
import cv2
import numpy as np
import joblib

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


DATASET = "dataset"

classes = ["glioma", "meningioma", "notumor", "pituitary"]

X = []
y = []


# Load images
for label in classes:

    folder = os.path.join(DATASET, label)

    print("Loading:", label)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        image = cv2.imread(path)

        if image is None:
            continue

        # Resize
        image = cv2.resize(image, (128, 128))

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize
        gray = gray / 255.0

        # Extract HOG features
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
        )

        X.append(features)
        y.append(label)


X = np.array(X)
y = np.array(y)

print("Total images:", len(X))


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create SVM
model = SVC(
    kernel="rbf",
    probability=True
)


# Train model
print("Training model...")

model.fit(X_train, y_train)


# Test model
predictions = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# Save model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/brain_tumor_svm.pkl")

print("\nModel saved successfully!")