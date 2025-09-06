import os
import shutil
import random

# Paths
source_dir = "Indian"
target_dir = "Dataset"

# Make target dataset folder
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Loop through each class folder (1-9, A-Z)
for folder in os.listdir(source_dir):
    class_source = os.path.join(source_dir, folder)
    class_target = os.path.join(target_dir, folder)

    if not os.path.isdir(class_source):
        continue

    # Create class folder in target
    if not os.path.exists(class_target):
        os.makedirs(class_target)

    # Get all images
    images = [f for f in os.listdir(class_source) if f.endswith(".jpg")]

    # Randomly sample 100 images
    selected = random.sample(images, 100)

    # Copy and rename them sequentially
    for idx, img in enumerate(selected):
        src_path = os.path.join(class_source, img)
        dst_path = os.path.join(class_target, f"{idx}.jpg")
        shutil.copy(src_path, dst_path)

    print(f"[INFO] Copied and renamed 100 images to {class_target}")

print("[DONE] Dataset created successfully.")
