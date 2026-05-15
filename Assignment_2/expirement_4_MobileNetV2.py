from cub_dataset import *
import numpy as np
import time
import cv2
import tensorflow as tf
import pandas as pd

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input
)
from sklearn.model_selection import KFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

OUTPUT_DIR = r'C:\Users\noahg\OneDrive\Desktop\Computer Vision\Assignment_2'

# Merge train and test data, then load bounding box images
merge_df = pd.concat([train_df, test_df], ignore_index=True)
X_all_bb, y_all_bb = load_images(merge_df, dataset_path, use_bbox=True)

print(f"X_all_bb shape: {X_all_bb.shape}")
print(f"y_all_bb shape: {y_all_bb.shape}")

X_all_bb = X_all_bb.astype("float32") / 255.0

# Convert class labels from 1-200 into 0-199
y_all_bb = y_all_bb - 1
num_classes = 200

# 5 different 80/20 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_losses     = []

for fold_no, (train_index, test_index) in enumerate(kf.split(X_all_bb), start=1):

    print(f"\n================ FOLD {fold_no} ================")

    X_train_bb = X_all_bb[train_index]
    X_test_bb  = X_all_bb[test_index]

    y_train_bb = y_all_bb[train_index]
    y_test_bb  = y_all_bb[test_index]

    print(f"X_train_bb shape: {X_train_bb.shape}")
    print(f"X_test_bb shape : {X_test_bb.shape}")
    print(f"y_train_bb shape: {y_train_bb.shape}")
    print(f"y_test_bb shape : {y_test_bb.shape}")

    # --- Build MobileNetV2 backbone ---
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(128, 128, 3)
    )

    # Freeze pretrained MobileNetV2
    base_model.trainable = False

    # MobileNetV2 outputs 1280 channels
    backbone_out = base_model.output

    # --- 1x1 Conv layers to preserve and compress MobileNetV2 features ---
    # Gradual reduction: 1280 -> 512 -> 256
    x = Conv2D(512, (1, 1), activation="relu", padding="same")(backbone_out)
    x = Conv2D(256, (1, 1), activation="relu", padding="same")(x)

    # --- 3x3 CNN layers to learn spatial features ---
    x = Conv2D(64,  (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Classification layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    # Create final model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    start_time = time.time()

    history = model.fit(
        X_train_bb,
        y_train_bb,
        epochs=25,
        batch_size=32,
        validation_data=(X_test_bb, y_test_bb)
    )

    end_time = time.time()
    print(f"\nTraining Time: {end_time - start_time:.2f} seconds")

    # Evaluate model
    print("\n================ RESULTS BOUNDING BOX ================")
    print(f"Final training accuracy   : {history.history['accuracy'][-1] * 100:.2f}%")
    print(f"Final validation accuracy : {history.history['val_accuracy'][-1] * 100:.2f}%")

    scores = model.evaluate(X_test_bb, y_test_bb, verbose=0)
    fold_losses.append(scores[0])
    fold_accuracies.append(scores[1])

    print(f"Fold {fold_no} loss     : {scores[0]:.4f}")
    print(f"Fold {fold_no} accuracy : {scores[1] * 100:.2f}%")

    print("\n================ CLASSIFICATION REPORT ================")
    predictions       = model.predict(X_test_bb)
    predicted_classes = np.argmax(predictions, axis=1)
    print(classification_report(y_test_bb, predicted_classes))

    print("\n================ CONFUSION MATRIX ================")
    cm = confusion_matrix(y_test_bb, predicted_classes)
    print(cm)

    # Save model per fold
    model_path = os.path.join(OUTPUT_DIR, f'mobilenet_model_fold{fold_no}.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Final 5-fold results
print("\n================ 5-FOLD 80/20 RESULTS ================")
print(f"Average loss     : {np.mean(fold_losses):.4f}")
print(f"Average accuracy : {np.mean(fold_accuracies) * 100:.2f}%")