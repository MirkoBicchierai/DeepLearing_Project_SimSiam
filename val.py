import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from DataLoader import ImageNetDataset
from Model import LinearEvaluationModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def accuracy(model, data_loader, device):
    correct = 0
    total = 0
    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():  # Disable gradient computation for evaluation
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            predicted = torch.argmax(y_pred, dim=1)  # Get the predicted class
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total

if "__main__" == __name__:

    train_dir = 'Dataset/SPLITTED/Train'
    val_dir = 'Dataset/SPLITTED/Test'

    batch_size = 48
    lr = 0.001
    epochs = 10

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

    model = torch.load("Models/model2.pth")
    model.stop_grad = True
    model.to(device)
    model.eval()

    LModel = LinearEvaluationModel(2048, val_dataset.num_classes, model)
    LModel.to(device)
    epochs = 10
    num_batches = len(train_loader)
    optimizer = optim.Adam(LModel.linear.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        LModel.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = LModel(x)
            loss = torch.nn.CrossEntropyLoss()(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / num_batches
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")

        acc = accuracy(LModel, val_loader, device)
        print(f"Epoch: {epoch}, Accuracy: {acc}")
