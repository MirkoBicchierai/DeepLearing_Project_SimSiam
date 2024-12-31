import math
import comet_ml
import torch
from torch.utils.data import DataLoader
from DataLoader import ImageNetDataset
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from Model import NetModel
from common import transformAug, transform, knn_validation


def lr_scheduling(opt, init_lr, actual_epoch, tot_epochs):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * actual_epoch / tot_epochs))
    for param_group in opt.param_groups:
        if 'fixed' in param_group and param_group['fixed']:
            param_group['fixed'] = init_lr
        else:
            param_group['fixed'] = cur_lr


def criterion(p, z, version='simplified'):
    if version == 'original': # paper implementation
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()
    elif version == 'simplified':  # same thing, much faster
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


if __name__ == "__main__":

    comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")

    train_dir = 'Dataset/SPLITTED/Train'
    val_dir = 'Dataset/SPLITTED/Test'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 48
    base_lr = 0.05
    lr = (base_lr * batch_size) / 256
    momentum = 0.9
    weight_decay = 0.0001
    epochs = 200
    test_step = 5
    knn_k = 200
    knn_t = 0.1

    exp = comet_ml.Experiment(
        project_name="Deep Learning Project",
        auto_metric_logging=False,
        auto_param_logging=False
    )
    parameters = {'batch_size': batch_size, 'learning_rate': lr, 'momentum': momentum, 'weight_decay': weight_decay}
    exp.log_parameters(parameters)

    train_dataset = ImageNetDataset(root_dir=train_dir, mode="train", transform=transformAug)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    train_dataset_evMode = ImageNetDataset(root_dir=train_dir, mode="eval", transform=transform)
    train_loader_ev = DataLoader(train_dataset_evMode, batch_size=batch_size, shuffle=False, num_workers=12,
                                 pin_memory=True)

    val_dataset = ImageNetDataset(root_dir=val_dir, mode="eval", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    model = NetModel(512, 128, stop_grad=True)
    model.to(device)

    optim_params = [{'params': model.encoder.parameters(), 'fixed': False},
                    {'params': model.predictor.parameters(), 'fixed': True}]
    optimizer = optim.SGD(optim_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler()

    num_batches = len(train_loader)
    for epoch in tqdm(range(epochs), desc="Training"):
        lr_scheduling(optimizer, lr, epoch, epochs)
        model.train()
        running_loss = 0
        all_normalized_outputs = []
        for images_aug1, images_aug2 in train_loader:
            images_aug1 = images_aug1.to(device)
            images_aug2 = images_aug2.to(device)
            optimizer.zero_grad()
            p1, p2, z1, z2 = model(images_aug1, images_aug2)
            loss = -(criterion(p1, z2) + criterion(p2, z1)) * 0.5
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            running_loss += loss.item()

            if epoch % test_step == 0:
                with torch.no_grad():
                    z1_norm = z1 / z1.norm(dim=1, keepdim=True)
                    all_normalized_outputs.append(z1_norm)
                    z2_norm = z2 / z2.norm(dim=1, keepdim=True)
                    all_normalized_outputs.append(z2_norm)

        epoch_loss = running_loss / num_batches
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")
        exp.log_metric('loss', epoch_loss, step=epoch)

        if epoch % test_step == 0:

            all_normalized_outputs = torch.cat(all_normalized_outputs, dim=0)
            std_per_channel = all_normalized_outputs.std(dim=0)  # Std per channel
            avg_epoch_std = std_per_channel.mean().item()  # Average

            top1_accuracy, top5_accuracy = knn_validation(model, train_loader_ev, val_loader,knn_k, knn_t, device)

            print(f"Epoch: {epoch + 1}, avg_std: {avg_epoch_std}, Top-1 Knn Accuracy: {top1_accuracy:.4f}, Top-5 Knn Accuracy: {top5_accuracy:.4f}")
            exp.log_metric('avg_std', avg_epoch_std, step=epoch)
            exp.log_metric('val_top1_accuracy', top1_accuracy, step=epoch)
            exp.log_metric('val_top5_accuracy', top5_accuracy, step=epoch)

        if epoch % 50 == 0:
            torch.save(model, "Models/SimSiam/model_" + str(epochs) + "_" + str(batch_size) + "_Checkpoint_" + str(epoch) + ".pth")

    torch.save(model, "Models/SimSiam/model_" + str(epochs) + "_" + str(batch_size) + ".pth")
