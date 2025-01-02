import torch
from torchvision.transforms import transforms
import torch.nn.functional as F

MiniImageNet_mean = [0.4727902,  0.44887177, 0.404713]
MiniImageNet_std = [0.28407582, 0.2758255,  0.29091981]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MiniImageNet_mean,  std=MiniImageNet_std)
])

transformAug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MiniImageNet_mean,  std=MiniImageNet_std),
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

def knn_validation(model, train_loader_ev, val_loader, knn_k, knn_t, device):
    model.eval()
    with torch.no_grad():

        top1_acc, top5_acc, total_num, feature_bank = 0, 0, 0, []
        classes = train_loader_ev.dataset.num_classes

        with torch.inference_mode():
            for data, target in train_loader_ev:
                feature = model.get_backbone_out(data.to(device))
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)

            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            feature_labels = torch.tensor(train_loader_ev.dataset.targets, device=feature_bank.device)

            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                feature = model.get_backbone_out(data)
                feature = F.normalize(feature, dim=1)
                pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, knn_k=knn_k, knn_t=knn_t)
                total_num += data.size(0)
                top1_acc += (pred_labels[:, 0] == target).float().sum().item()
                top5_acc += sum([target[i].item() in pred_labels[i, :5].tolist() for i in range(target.size(0))])

        top1_acc = top1_acc / total_num
        top5_acc = top5_acc / total_num

    return top1_acc, top5_acc

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k=200, knn_t=0.1):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


