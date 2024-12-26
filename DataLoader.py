import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

transformAug = T.Compose([
    T.RandomResizedCrop(size=224, scale=(0.2, 1.0)),  # Scale range [0.2, 1.0]
    T.RandomHorizontalFlip(),  # Horizontal flip with 50% probability
    T.RandomApply([  # Apply color jitter with a probability of 0.8
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], p=0.8),
    T.RandomGrayscale(p=0.2),  # 20% chance to convert to grayscale
    T.RandomApply([  # Apply Gaussian blur with std in [0.1, 2.0]
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.5)
])

class ImageNetDataset(Dataset):
    def __init__(self, root_dir,mode, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.num_classes = 0
        self.mode = mode

        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.num_classes += 1
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.mode == "train":
            # Return two augmentations of the same image for self-supervised training
            aug1 = transformAug(image)
            aug2 = transformAug(image)
            return aug1, aug2

        elif self.mode == "eval":
            # Return a single normalized image with its label for evaluation
            return image, label
