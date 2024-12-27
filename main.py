import comet_ml
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from DataLoader import ImageNetDataset
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from Model import NetModel
from common import transformAug, transform


def criterion(p, z):
    return F.cosine_similarity(p, z, dim=-1).mean()


def extract_features(m, dataloader, device):
    m.eval()
    features = []
    labels = []
    with torch.inference_mode():
        for images, targets in dataloader:
            images = images.to(device)
            out = m.f(images)
            features.append(out.cpu().numpy())
            labels.append(targets.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


if __name__ == "__main__":

    comet_ml.login()

    train_dir = 'Dataset/SPLITTED/Train'
    val_dir = 'Dataset/SPLITTED/Test'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 48
    base_lr = 0.05
    lr = (base_lr*batch_size)/256
    momentum = 0.9
    weight_decay = 0.0001
    epochs = 100

    train_dataset = ImageNetDataset(root_dir=train_dir, mode="train", transform=transformAug)
    train_dataset_evMode = ImageNetDataset(root_dir=train_dir, mode="eval", transform=transform)
    val_dataset = ImageNetDataset(root_dir=val_dir, mode="eval", transform=transform)

    train_loader_ev = DataLoader(train_dataset_evMode, batch_size=batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    exp = comet_ml.Experiment(
        project_name="Deep Learning Project",
        auto_metric_logging=False,
        auto_param_logging=False
    )
    parameters = {'batch_size': batch_size, 'learning_rate': lr, 'momentum': momentum, 'weight_decay': weight_decay}
    exp.log_parameters(parameters)

    model = NetModel(2048, 512, stop_grad=True)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    num_batches = len(train_loader)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0
        all_normalized_outputs = []
        for images_aug1, images_aug2 in train_loader:
            images_aug1 = images_aug1.to(device)
            images_aug2 = images_aug2.to(device)
            p1, p2, z1, z2 = model(images_aug1, images_aug2)
            loss = -(criterion(p1, z2) / 2 + criterion(p2, z1) / 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            with torch.no_grad():
                # L2 Normalization for z1 and z2
                z1_norm = z1 / z1.norm(dim=1, keepdim=True)
                all_normalized_outputs.append(z1_norm)
                z2_norm = z2 / z2.norm(dim=1, keepdim=True)
                all_normalized_outputs.append(z2_norm)

        scheduler.step()

        epoch_loss = running_loss / num_batches
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")
        exp.log_metric('loss', epoch_loss, step=epoch)

        # Std per channel and Average
        all_normalized_outputs = torch.cat(all_normalized_outputs, dim=0)
        std_per_channel = all_normalized_outputs.std(dim=0)  # Std per channel
        avg_epoch_std = std_per_channel.mean().item()  # Average
        print(f"Epoch: {epoch + 1}, avg_std: {avg_epoch_std}")
        exp.log_metric('avg_std', avg_epoch_std, step=epoch)

        model.eval()
        with torch.no_grad():
            train_features, train_labels = extract_features(model, train_loader_ev, device)
            val_features, val_labels = extract_features(model, val_loader, device)

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(train_features, train_labels)

            val_probs = knn.predict_proba(val_features)
            val_probs_tensor = torch.tensor(val_probs)

            # Top-1 accuracy
            _, top1_pred = torch.topk(val_probs_tensor, k=1, dim=1)
            top1_accuracy = sum(val_labels == top1_pred.squeeze()) / len(val_labels)

            # Top-5 accuracy
            _, top5_pred = torch.topk(val_probs_tensor, k=5, dim=1)
            top5_accuracy = sum(any(pred == label for pred in top5_pred[i])
                                for i, label in enumerate(val_labels)) / len(val_labels)

            print(f"Epoch: {epoch + 1}, Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}")
            exp.log_metric('val_top1_accuracy', top1_accuracy, step=epoch)
            exp.log_metric('val_top5_accuracy', top5_accuracy, step=epoch)

    torch.save(model, "Models/SimSiam/model_" + str(epochs) + "_" + str(batch_size) + "_" + str(lr) + ".pth")
