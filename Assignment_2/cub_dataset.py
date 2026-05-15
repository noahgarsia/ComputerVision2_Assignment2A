import os
import pandas as pd
import cv2
import numpy as np


# Load the dataset metadata
# dataset_path = r"C:\Users\noahg\Downloads\CUB_200_2011\CUB_200_2011\CUB_200_2011"
dataset_path = r"C:\Users\noahg\Downloads\CUB_200_2011_Subset20classes\CUB_200_2011_Subset20classes"
images_path = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                          sep=' ', header=None, names=['img_id', 'filepath'])


image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                 sep=' ', header=None, names=['img_id', 'class_id'])


bounding_boxes = pd.read_csv(os.path.join(dataset_path, 'bounding_boxes.txt'),
                             sep=' ', header=None, names=['img_id', 'x', 'y', 'width', 'height'])


split_training_testing = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                     sep=' ', header=None, names=['img_id', 'is_train'])


# Merge all 4 dataframes into one table using img_id as the common key
df = images_path.merge(image_class_labels, on='img_id')
df = df.merge(bounding_boxes,         on='img_id')
df = df.merge(split_training_testing, on='img_id')


train_df = df[df['is_train'] == 1]
test_df  = df[df['is_train'] == 0]

# print(f'Training: {len(train_df)} | Testing: {len(test_df)}')


# Load images from disk
# use_bbox=False loads the whole image (Experiment 1 and 3)
# use_bbox=True  crops to the bounding box region only (Experiment 2 and 4)
def load_images(df, dataset_path, use_bbox=False, image_size=(128, 128)):
    images = []
    labels = []

    for _, row in df.iterrows():
        # Build full path and read image from disk
        img_path = os.path.join(dataset_path, 'images', row['filepath'])
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        # OpenCV reads as BGR by default, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop to bounding box only if use_bbox is True
        if use_bbox:
            x, y, w, h = int(row['x']), int(row['y']), int(row['width']), int(row['height'])
            img = img[y:y+h, x:x+w]

        # Resize so all images are the same dimensions
        img = cv2.resize(img, image_size)
        images.append(img)
        labels.append(row['class_id'])

    return np.array(images), np.array(labels)


# Load training and testing images
# X_train, y_train = load_images(train_df, dataset_path, use_bbox=False)
# X_test,  y_test  = load_images(test_df,  dataset_path, use_bbox=False)
# 
# print(f"X_train shape: {X_train.shape}")
# print(f"X_test shape : {X_test.shape}")