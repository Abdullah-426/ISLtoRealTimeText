import numpy as np
import cv2
import os
from image_processing import func

# Input dataset (raw captured images)
input_path = "Dataset"

# Output dataset (processed and split)
output_path = "smaller-data"
train_path = os.path.join(output_path, "train")
test_path = os.path.join(output_path, "test")

# Create directories if not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

var = 0   # total images
c1 = 0    # train images
c2 = 0    # test images

# Walk through each class folder
for (dirpath, dirnames, filenames) in os.walk(input_path):
    for dirname in dirnames:
        print("Processing:", dirname)

        # Paths for this class
        class_input_path = os.path.join(input_path, dirname)
        train_class_path = os.path.join(train_path, dirname)
        test_class_path = os.path.join(test_path, dirname)

        # Create train/test subfolders for this class
        if not os.path.exists(train_class_path):
            os.makedirs(train_class_path)
        if not os.path.exists(test_class_path):
            os.makedirs(test_class_path)

        # All files for this class
        files = os.listdir(class_input_path)

        # Train/Test split: 75% train, 25% test
        num = int(0.75 * len(files))
        i = 0

        for file in files:
            var += 1
            actual_path = os.path.join(class_input_path, file)

            # Process image
            bw_image = func(actual_path)

            if i < num:  # Train set
                c1 += 1
                save_path = os.path.join(train_class_path, file)
                cv2.imwrite(save_path, bw_image)
            else:        # Test set
                c2 += 1
                save_path = os.path.join(test_class_path, file)
                cv2.imwrite(save_path, bw_image)

            i += 1

print("Total images processed:", var)
print("Train images:", c1)
print("Test images:", c2)
