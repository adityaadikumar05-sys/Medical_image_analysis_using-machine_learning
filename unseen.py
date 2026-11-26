import os
import cv2
import joblib

from skimage.feature import hog


# ==========================================
# 1. LOAD TRAINED SVM MODEL
# ==========================================

model = joblib.load("model/brain_tumor_svm.pkl")

print("\n====================================")
print("   BRAIN TUMOR DETECTION SYSTEM")
print("             HOG + SVM")
print("====================================")

print("\nTrained model loaded successfully!")

print("\nClasses:", model.classes_)
print("Kernel :", model.kernel)


# ==========================================
# 2. FUNCTION FOR PREPROCESSING + HOG
# ==========================================

def extract_features(image):

    # Resize image
    image = cv2.resize(image, (128, 128))

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Normalize pixel values
    gray = gray / 255.0

    # Extract HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    )

    return features


# ==========================================
# 3. UNSEEN DATA FOLDER
# ==========================================

UNSEEN_FOLDER = "unseen"

print("\n====================================")
print("      UNSEEN DATA PREDICTION")
print("====================================")


# ==========================================
# 4. READ ALL UNSEEN IMAGES
# ==========================================

for filename in sorted(os.listdir(UNSEEN_FOLDER)):

    path = os.path.join(
        UNSEEN_FOLDER,
        filename
    )

    # Read image
    image = cv2.imread(path)

    # Skip non-image files
    if image is None:
        continue


    # ======================================
    # 5. EXTRACT HOG FEATURES
    # ======================================

    features = extract_features(image)


    # ======================================
    # 6. PREDICT USING TRAINED SVM
    # ======================================

    prediction = model.predict(
        [features]
    )[0]


    # Get probability
    probabilities = model.predict_proba(
        [features]
    )[0]

    confidence = max(probabilities) * 100


    # ======================================
    # 7. DISPLAY RESULT
    # ======================================

    print("\n------------------------------------")

    print("Image      :", filename)

    print(
        "Prediction :",
        prediction.upper()
    )

    print(
        "Confidence :",
        round(confidence, 2),
        "%"
    )

    print("\nClass Probabilities:")

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):

        print(
            " ",
            class_name,
            ":",
            round(probability * 100, 2),
            "%"
        )


print("\n====================================")
print("      PREDICTION COMPLETED")
print("====================================")