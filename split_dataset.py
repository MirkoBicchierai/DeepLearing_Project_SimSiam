import os
import random
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def split_dataset(root_dir, train_ratio=0.8):
    train_dir = "Dataset/SPLITTED/Train"
    test_dir = "Dataset/SPLITTED/Test"

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        file_names = os.listdir(class_path)
        random.shuffle(file_names)

        split_index = int(len(file_names) * train_ratio)
        train_files = file_names[:split_index]
        test_files = file_names[split_index:]

        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir)
        os.makedirs(test_class_dir)

        for file_name in train_files:
            src = os.path.join(class_path, file_name)
            dst = os.path.join(train_class_dir, file_name)
            shutil.copy(src, dst)

        for file_name in test_files:
            src = os.path.join(class_path, file_name)
            dst = os.path.join(test_class_dir, file_name)
            shutil.copy(src, dst)

        print(f"Processed class '{class_name}' with {len(train_files)} train and {len(test_files)} test files.")


def mean_std_dataset(dataset_path):
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    num_pixels = 0

    for class_folder in tqdm(os.listdir(dataset_path), desc="Processing classes"):
        class_folder_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_folder_path):
            continue

        for image_name in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_name)
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_np = np.array(img) / 255.0
                pixel_sum += img_np.sum(axis=(0, 1))
                pixel_sq_sum += (img_np ** 2).sum(axis=(0, 1))
                num_pixels += img_np.shape[0] * img_np.shape[1]

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)
    return mean, std


if __name__ == "__main__":
    root_folder = "Dataset/CLEAR"
    split_dataset(root_folder)
    m, s = mean_std_dataset(root_folder)
    print(f"Mean: {m}")
    print(f"Standard Deviation: {s}")
