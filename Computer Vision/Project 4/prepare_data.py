import numpy as np
import cv2
import os
import random

# Define paths for YOLOv5 images and labels
yolo_train_images_dir = "train/images"
yolo_train_labels_dir = "train/labels"
yolo_valid_images_dir = "valid/images"
yolo_valid_labels_dir = "valid/labels"

# Define paths for U-Net images and masks
unet_images_dir = "unet/images"
unet_masks_dir = "unet/masks"
unet_train_images_dir = "unet/train/images"
unet_train_masks_dir = "unet/train/masks"
unet_valid_images_dir = "unet/valid/images"
unet_valid_masks_dir = "unet/valid/masks"

# Create directories if they don’t exist
os.makedirs(yolo_train_images_dir, exist_ok=True)
os.makedirs(yolo_train_labels_dir, exist_ok=True)
os.makedirs(yolo_valid_images_dir, exist_ok=True)
os.makedirs(yolo_valid_labels_dir, exist_ok=True)
os.makedirs(unet_train_images_dir, exist_ok=True)
os.makedirs(unet_train_masks_dir, exist_ok=True)
os.makedirs(unet_valid_images_dir, exist_ok=True)
os.makedirs(unet_valid_masks_dir, exist_ok=True)

# Load the dataset
data = np.load("train.npz")
images = data['images']
labels = data['labels']
bboxes = data['bboxes']
semantic_masks = data['semantic_masks']

# Split data (20% validation)
split_ratio = 0.2
num_valid = int(len(images) * split_ratio)
valid_indices = random.sample(range(len(images)), num_valid)

def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [x_center, y_center, width, height]

for i, img in enumerate(images):
    img = img.reshape(64, 64, 3)
    mask = semantic_masks[i].reshape(64, 64)
    img_filename = f"{i}.jpg"
    mask_filename = f"{i}.png"
    
    if i in valid_indices:
        # Validation set
        cv2.imwrite(os.path.join(yolo_valid_images_dir, img_filename), img)
        cv2.imwrite(os.path.join(unet_valid_images_dir, img_filename), img)
        cv2.imwrite(os.path.join(unet_valid_masks_dir, mask_filename), mask)
        with open(os.path.join(yolo_valid_labels_dir, f"{i}.txt"), 'w') as f:
            for j in range(2):
                bbox = convert_bbox_to_yolo_format(bboxes[i, j], 64, 64)
                f.write(f"{labels[i, j]} {' '.join(map(str, bbox))}\n")
    else:
        # Training set
        cv2.imwrite(os.path.join(yolo_train_images_dir, img_filename), img)
        cv2.imwrite(os.path.join(unet_train_images_dir, img_filename), img)
        cv2.imwrite(os.path.join(unet_train_masks_dir, mask_filename), mask)
        with open(os.path.join(yolo_train_labels_dir, f"{i}.txt"), 'w') as f:
            for j in range(2):
                bbox = convert_bbox_to_yolo_format(bboxes[i, j], 64, 64)
                f.write(f"{labels[i, j]} {' '.join(map(str, bbox))}\n")

print(f"Dataset split completed: {num_valid} images in validation set.")
