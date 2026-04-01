import os
import shutil
import random

# Paths
source_dir = "PetImages"   # folder from Microsoft dataset
base_dir = "data"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

classes = ["Cat", "Dog"]

# Create folders
for split in [train_dir, val_dir]:
    for cls in classes:
        os.makedirs(os.path.join(split, cls.lower()), exist_ok=True)

# Split ratio
split_ratio = 0.8

for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)

    # Remove corrupt images (important)
    images = [img for img in images if img.endswith(('.jpg', '.png'))]

    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Copy files
    for img in train_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(train_dir, cls.lower(), img)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

    for img in val_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(val_dir, cls.lower(), img)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

print("Dataset split complete!")