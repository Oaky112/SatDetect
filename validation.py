import os
import shutil
import random

# Define directories
train_dir = "SCSF/train"
validation_dir = "SCSF/validation"

# Create validation directory if not exists
os.makedirs(validation_dir, exist_ok=True)

# Get list of image filenames in the training directory
image_filenames = os.listdir(os.path.join(train_dir, "images"))

# Calculate number of images for validation set (10% of training set)
num_validation_images = int(len(image_filenames) * 0.1)

# Randomly select images for validation
validation_images = random.sample(image_filenames, num_validation_images)

# Move validation images to validation directory
for image in validation_images:
    image_path = os.path.join(train_dir, "images", image)
    label_path = os.path.join(train_dir, "labels", image.replace(".jpg", ".txt"))
    shutil.move(image_path, os.path.join(validation_dir, "images"))
    shutil.move(label_path, os.path.join(validation_dir, "labels"))

print("Validation set created successfully.")



