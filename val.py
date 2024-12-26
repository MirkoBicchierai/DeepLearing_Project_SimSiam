import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from DataLoader import ImageNetDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get the top-k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # Transpose the predictions tensor
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # Check if predictions match the targets

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Count correct predictions
            res.append(correct_k.mul_(100.0 / batch_size))  # Compute accuracy
        return res


def validate(val_loader, model):
    model.eval()

    top1_acc = 0.0
    top5_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(device)
            target = target.to(device)

            # Forward pass
            output = model.singleImage(images)  # Use the model's forward method, not `singleImage`

            # Compute top-1 and top-5 accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # Accumulate accuracy
            top1_acc += acc1[0] * images.size(0)
            top5_acc += acc5[0] * images.size(0)
            total_samples += images.size(0)

    # Compute average accuracy
    top1_acc /= total_samples
    top5_acc /= total_samples

    return top1_acc, top5_acc


if "__main__" == __name__:
    train_dir = 'Dataset/SPLITTED/Train'
    val_dir = 'Dataset/SPLITTED/Test'

    batch_size = 48
    base_lr = 0.05
    lr = (base_lr * batch_size) / 256
    momentum = 0.9
    weight_decay = 0.0001
    epochs = 20

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                             std=[0.229, 0.224, 0.225])  # Normalize with ImageNet std
    ])

    train_dataset = ImageNetDataset(root_dir=train_dir, mode="eval", transform=transform)
    val_dataset = ImageNetDataset(root_dir=val_dir, mode="eval", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = torch.load("Models/model.pth")
    model.to(device)
    model.eval()

    # Evaluate the model on the validation set
    acc1, acc5 = validate(val_loader, model)

    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-5 Accuracy: {acc5:.2f}%")
