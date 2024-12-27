from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                         std=[0.229, 0.224, 0.225])  # Normalize with ImageNet std
])

transformAug = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),  # Scale range [0.2, 1.0]
    transforms.RandomHorizontalFlip(),  # Horizontal flip with 50% probability
    transforms.RandomApply([  # Apply color jitter with a probability of 0.8
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),  # 20% chance to convert to grayscale
    transforms.RandomApply([  # Apply Gaussian blur with std in [0.1, 2.0]
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.5)
])