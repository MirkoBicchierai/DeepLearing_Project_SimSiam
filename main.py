import comet_ml
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from DataLoader import ImageNetDataset
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from Model import NetModel


def criterion(p, z):
    return F.cosine_similarity(p, z, dim=-1).mean()


if __name__ == "__main__":

    comet_ml.login()

    train_dir = 'Dataset/SPLITTED/Train'
    val_dir = 'Dataset/SPLITTED/Test'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 48
    base_lr = 0.05
    lr = (base_lr * batch_size) / 256  # (base_lr*batch_size)/256
    momentum = 0.9
    weight_decay = 0.0001
    epochs = 100

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                             std=[0.229, 0.224, 0.225])  # Normalize with ImageNet std
    ])

    train_dataset = ImageNetDataset(root_dir=train_dir, mode="train", transform=transform)
    val_dataset = ImageNetDataset(root_dir=val_dir, mode="eval", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    exp = comet_ml.Experiment(
        project_name="Deep Learning Project",
        auto_metric_logging=False,
        auto_param_logging=False
    )
    parameters = {'batch_size': batch_size, 'learning_rate': lr, 'momentum': momentum, 'weight_decay': weight_decay}
    exp.log_parameters(parameters)

    f = NetModel(2048, 512, stop_grad=True)
    f.to(device)

    optimizer = optim.SGD(f.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    num_batches = len(train_loader)

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for images_aug1, images_aug2 in train_loader:
            images_aug1 = images_aug1.to(device)
            images_aug2 = images_aug2.to(device)
            p1, p2, z1, z2 = f(images_aug1, images_aug2)
            loss = -(criterion(p1, z2) / 2 + criterion(p2, z1) / 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / num_batches
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")
        exp.log_metric('loss', epoch_loss, step=epoch)

    torch.save(f, "Models/SimSiam/model_" + str(epochs) + "_" + str(batch_size) + "_" + str(lr) + ".pth")
