# Medical_image_analysis_using-machine_learning
Brain Tumor Classification using traditional Machine Learning with HOG feature extraction and SVM. Classifies brain MRI images into Glioma, Meningioma, Pituitary Tumor, and No Tumor using Python, OpenCV, scikit-image, and scikit-learn.


# Brain Tumor Classification Using HOG + SVM

This project implements a **Brain Tumor MRI Classification System using traditional Machine Learning**, without using CNNs or Deep Learning. The system classifies brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The project uses **Histogram of Oriented Gradients (HOG)** for feature extraction and a **Support Vector Machine (SVM)** with an RBF kernel for classification.

## How It Works

The classification pipeline follows these steps:

**MRI Image → Resize → Grayscale → Normalization → HOG Feature Extraction → SVM → Prediction**

MRI images are first resized to **128 × 128 pixels**, converted to grayscale, and normalized. HOG then extracts gradient and edge-based features from each image and converts them into numerical feature vectors.

These feature vectors are used to train an SVM classifier. The dataset is divided into **80% training data and 20% testing data** using a stratified train-test split.

The trained model can classify previously unseen MRI images into one of the four supported classes and display prediction probabilities for each class.

## Technologies Used

- Python
- OpenCV
- NumPy
- scikit-image
- scikit-learn
- Joblib

## Machine Learning Pipeline

1. Load MRI images from the dataset.
2. Resize each image to 128 × 128 pixels.
3. Convert the image to grayscale.
4. Normalize pixel values.
5. Extract HOG features.
6. Split the dataset into training and testing sets.
7. Train an SVM classifier using the RBF kernel.
8. Evaluate the model using accuracy, precision, recall, and F1-score.
9. Save the trained model using Joblib.
10. Use the saved model to classify unseen MRI images.

## HOG Feature Extraction

HOG (Histogram of Oriented Gradients) extracts information about local intensity gradients, edges, and their orientations from MRI images.

The project uses:

- 9 gradient orientations
- 8 × 8 pixels per cell
- 2 × 2 cells per block

HOG converts each MRI image into a fixed-length numerical feature vector that can be provided to the SVM classifier.

## Support Vector Machine

Support Vector Machine is a supervised Machine Learning algorithm. This project uses an **RBF kernel**, allowing the SVM to model nonlinear relationships between the HOG features of different MRI classes.

The problem is a **four-class supervised classification problem**.

## Unseen Data Prediction

After training, the SVM model is saved as a `.pkl` file. New MRI images can then be processed using the same preprocessing and HOG configuration and classified without retraining the model.

In the project demonstration, the model successfully classified unseen examples from all four supported categories.

## Project Structure

```text
Brain-Tumor-Detection/
│
├── dataset/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── unseen/
│
├── model/
│   └── brain_tumor_svm.pkl
│
├── tumor_detection.py
├── predict_unseen.py
└── README.md
```

## Purpose

The primary purpose of this project is to demonstrate how traditional Machine Learning and handcrafted image features can be applied to medical-image classification without relying on CNNs or other Deep Learning architectures.

> **Note:** This project is intended for educational and academic purposes. It is an MRI image classification system and should not be considered a substitute for professional medical diagnosis.
