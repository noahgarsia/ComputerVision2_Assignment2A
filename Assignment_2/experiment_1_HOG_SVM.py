from cub_dataset import *

import numpy as np
import time
import sys
import joblib
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Load whole images for Experiment 1
X_train, y_train = load_images(train_df, dataset_path, use_bbox=True)
X_test,  y_test  = load_images(test_df,  dataset_path, use_bbox=True)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape : {y_test.shape}")

# Function to extract hog features
def extract_hog_features(images):
    features = []

    for img in images:
        hog_features = hog(
            img,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            channel_axis=-1
        )
        features.append(hog_features)

    return np.array(features)

# Extract HOG features for training and testing sets
start_hog_train = time.time()
X_train_hog = extract_hog_features(X_train)
hog_train_time = time.time() - start_hog_train

start_hog_test = time.time()
X_test_hog = extract_hog_features(X_test)
hog_test_time = time.time() - start_hog_test

print(f"X_train_hog shape: {X_train_hog.shape}")
print(f"X_test_hog shape : {X_test_hog.shape}")

print(f"HOG train extraction time: {hog_train_time:.2f} seconds")
print(f"HOG test extraction time : {hog_test_time:.2f} seconds")

# Building SVM
svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0)
)

# Training SVM
start_train = time.time()
svm.fit(X_train_hog, y_train)
training_time = time.time() - start_train

print("SVM training complete!")
print(f"SVM training time: {training_time:.2f} seconds")

# Evaluation
train_predictions = svm.predict(X_train_hog)
test_predictions  = svm.predict(X_test_hog)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy  = accuracy_score(y_test,  test_predictions)

print("\n================ RESULTS ================")

print(f"Training accuracy : {train_accuracy * 100:.2f}%")
print(f"Testing accuracy  : {test_accuracy * 100:.2f}%")

print("\n================ CLASSIFICATION REPORT ================")
print(classification_report(y_test, test_predictions))

# Save the model
joblib.dump(svm, r'C:\Users\noahg\OneDrive\Desktop\Computer Vision\Assignment_2\svm_model.pkl')
print("Model saved!")

# Per-class F1 score bar chart
f1_scores = f1_score(y_test, test_predictions, average=None, labels=np.unique(y_test))
classes = np.unique(y_test)

plt.figure(figsize=(20, 6))
plt.bar(classes, f1_scores, color='steelblue', edgecolor='black', linewidth=0.5)
plt.xlabel('Class', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.title('Experiment 1: HOG + SVM — Per-Class F1 Score', fontsize=14)
plt.xticks(classes, rotation=90, fontsize=7)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(r'C:\Users\noahg\OneDrive\Desktop\Computer Vision\Assignment_2\experiment_1_f1_per_class.png', dpi=150)
plt.show()
print("F1 chart saved!")