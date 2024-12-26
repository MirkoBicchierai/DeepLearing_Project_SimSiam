import os
import random
import shutil

def split_dataset(root_dir, train_ratio=0.8):
    train_dir = "Dataset/SPLITTED/Train"
    test_dir = "Dataset/SPLITTED/Test"

    # Iterate through each class folder
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)

        # Skip if not a directory (e.g., train/test folders themselves)
        if not os.path.isdir(class_path):
            continue

        # Get all file names in the class folder
        file_names = os.listdir(class_path)
        random.shuffle(file_names)  # Shuffle the files

        # Split files into train and test sets
        split_index = int(len(file_names) * train_ratio)
        train_files = file_names[:split_index]
        test_files = file_names[split_index:]

        # Create corresponding class subfolders in train and test directories
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir)
        os.makedirs(test_class_dir)

        # Move files to train and test directories
        for file_name in train_files:
            src = os.path.join(class_path, file_name)
            dst = os.path.join(train_class_dir, file_name)
            shutil.copy(src, dst)

        for file_name in test_files:
            src = os.path.join(class_path, file_name)
            dst = os.path.join(test_class_dir, file_name)
            shutil.copy(src, dst)

        print(f"Processed class '{class_name}' with {len(train_files)} train and {len(test_files)} test files.")

if __name__ == "__main__":
    root_folder = "/home/mirko/PycharmProjects/DeepLearing Project/Dataset/CLEAR"  # Change this to your root dataset folder
    split_dataset(root_folder)
