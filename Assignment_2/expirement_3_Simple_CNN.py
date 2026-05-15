from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold
from cub_dataset import *
import numpy as np
import time
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import matplotlib.pyplot as plt

#Location to save models and charts on my local machine
OUTPUT_DIR = r'C:\Users\noahg\OneDrive\Desktop\Computer Vision\Assignment_2'

#Merge train and test data for K-Fold
X_all, y_all = load_images(pd.concat([train_df, test_df]).reset_index(drop=True),
                           dataset_path, use_bbox=True)

print(f"X_all shape: {X_all.shape}")
print(f"y_all shape: {y_all.shape}")

X_all = X_all.astype("float32") / 255.0

# Convert class labels from 1-200 into 0-199
y_all = y_all - 1
num_classes = 200

# 5 different 80/20 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_losses     = []

for fold_no, (train_index, test_index) in enumerate(kf.split(X_all, y_all), start=1):
    if fold_no != 1:
        continue  # only run the first fold 

    print(f"\n================ FOLD {fold_no} / 5 ================")

    X_train, X_val = X_all[train_index], X_all[test_index]
    y_train, y_val = y_all[train_index], y_all[test_index]

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape  : {X_val.shape}")

    # Build model 
    model = Sequential()

    # CNN Layer 1
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # CNN Layer 2
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # CNN Layer 3
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Classification layers
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_data=(X_val, y_val)
    )

    end_time = time.time()
    print(f"\nTraining Time: {end_time - start_time:.2f} seconds")

    fold_accuracies.append(history.history['val_accuracy'][-1])
    fold_losses.append(history.history['val_loss'][-1])

    print("\n================ RESULTS BOUNDING BOX ================")
    print(f"Final training accuracy   : {history.history['accuracy'][-1]    * 100:.2f}%")
    print(f"Final validation accuracy : {history.history['val_accuracy'][-1] * 100:.2f}%")

    print("\n================ CLASSIFICATION REPORT ================")
    predictions       = model.predict(X_val)
    predicted_classes = np.argmax(predictions, axis=1)
    print(classification_report(y_val, predicted_classes))

    model_path = os.path.join(OUTPUT_DIR, 'cnn_model_fold1.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    f1_scores = f1_score(y_val, predicted_classes, average=None, labels=np.unique(y_val))
    classes   = np.unique(y_val)

    plt.figure(figsize=(20, 6))
    plt.bar(classes, f1_scores, color='steelblue', edgecolor='black', linewidth=0.5)
    plt.xlabel('Class',    fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Experiment 3: CNN — Per-Class F1 Score (Fold 1)', fontsize=14)
    plt.xticks(classes, rotation=90, fontsize=7)
    plt.ylim(0, 1)
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, 'experiment_3_f1_per_class_fold1_f1.png')
    plt.savefig(chart_path, dpi=150)
    plt.show()
    print(f"F1 chart saved to {chart_path}")