import joblib

from cub_dataset import *
import numpy as np
import time
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.cluster import KMeans


# Load bounding box images
X_train_bb, y_train_bb = load_images(train_df, dataset_path, use_bbox=True)
X_test_bb,  y_test_bb  = load_images(test_df,  dataset_path, use_bbox=True)


print(f"X_train_bb shape: {X_train_bb.shape}")
print(f"X_test_bb shape : {X_test_bb.shape}")
print(f"y_train_bb shape: {y_train_bb.shape}")
print(f"y_test_bb shape : {y_test_bb.shape}")


# Extract SIFT descriptors
sift = cv2.SIFT_create()


def extract_sift_descriptors(images):
    descriptors_list = []

    for img in images:
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(grey, None)

        if descriptors is None:
            descriptors = np.zeros((1, 128))

        descriptors_list.append(descriptors)

    return descriptors_list


start_sift_train = time.time()
train_descriptors_bb = extract_sift_descriptors(X_train_bb)
sift_train_time = time.time() - start_sift_train

start_sift_test = time.time()
test_descriptors_bb  = extract_sift_descriptors(X_test_bb)
sift_test_time = time.time() - start_sift_test

print(f"SIFT train extraction time: {sift_train_time:.2f} seconds")
print(f"SIFT test extraction time : {sift_test_time:.2f} seconds")


# Bag of Visual Words
def build_bag_of_words(train_descriptors, test_descriptors, num_clusters=100):
    all_descriptors = np.vstack(train_descriptors)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)

    def create_histograms(descriptor_list):
        histograms = []
        for descriptors in descriptor_list:
            cluster_labels = kmeans.predict(descriptors)
            histogram, _ = np.histogram(cluster_labels, bins=np.arange(num_clusters + 1))
            histograms.append(histogram)
        return np.array(histograms)

    X_train_bow = create_histograms(train_descriptors)
    X_test_bow  = create_histograms(test_descriptors)

    return X_train_bow, X_test_bow


X_train_bow_bb, X_test_bow_bb = build_bag_of_words(
    train_descriptors_bb,
    test_descriptors_bb
)

print(f"X_train_bow_bb shape : {X_train_bow_bb.shape}")
print(f"X_test_bow_bb shape  : {X_test_bow_bb.shape}")


# Build Random Forest classifier
rf_bb = RandomForestClassifier(n_estimators=20, random_state=42)


# Train Random Forest
start_train_bb = time.time()
rf_bb.fit(X_train_bow_bb, y_train_bb)
training_time_bb = time.time() - start_train_bb

print("Random Forest training complete!")
print(f"Training time bbox        : {training_time_bb:.2f} seconds")


# Evaluate model
train_predictions_bb = rf_bb.predict(X_train_bow_bb)
test_predictions_bb  = rf_bb.predict(X_test_bow_bb)

train_accuracy_bb = accuracy_score(y_train_bb, train_predictions_bb)
test_accuracy_bb  = accuracy_score(y_test_bb,  test_predictions_bb)

print("\n================ RESULTS BOUNDING BOX ================")
print(f"Training accuracy : {train_accuracy_bb * 100:.2f}%")
print(f"Testing accuracy  : {test_accuracy_bb  * 100:.2f}%")

print("\n================ CLASSIFICATION REPORT BOUNDING BOX ================")
print(classification_report(y_test_bb, test_predictions_bb))

print("\n================ CONFUSION MATRIX ================")
cm = confusion_matrix(y_test_bb, test_predictions_bb)
print(cm)



# Save the model
joblib.dump(rf_bb, r'C:\Users\noahg\OneDrive\Desktop\Computer Vision\Assignment_2\rf_bb_model.pkl')
print("Model saved!")