import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader import ImageNetDataset
from Model import LinearEvaluationModel
from common import transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy(model, data_loader, device):
    top1_correct = 0
    top5_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            total += y.size(0)

            # Top-1 accuracy
            top1_predicted = torch.argmax(y_pred, dim=1)
            top1_correct += (top1_predicted == y).sum().item()

            # Top-5 accuracy
            _, top5_predicted = torch.topk(y_pred, 5, dim=1)
            top5_correct += sum([y[i] in top5_predicted[i] for i in range(len(y))])

    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total

    return top1_accuracy, top5_accuracy


if "__main__" == __name__:

    dataset = "MiniImageNet"

    train_dir = 'Dataset/SPLITTED/Train'
    val_dir = 'Dataset/SPLITTED/Test'

    batch_size = 48
    lr = 0.001
    epochs = 10

    train_dataset = ImageNetDataset(root_dir=train_dir, mode="eval", transform=transform)
    val_dataset = ImageNetDataset(root_dir=val_dir, mode="eval", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = torch.load("Models/SimSiam/model100.pth")
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
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")

        acc_1, acc_5 = accuracy(LModel, val_loader, device)
        print(f"Epoch: {epoch + 1}, Accuracy top1: {acc_1}, Accuracy top5: {acc_5}")

    torch.save(LModel, "Models/LinearEval/model_" + dataset + "_" + str(epochs) + "_" + str(batch_size) + "_" + str(
        lr) + ".pth")
