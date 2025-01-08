import math
import comet_ml
import torch
from torch.utils.data import DataLoader
from DataLoader import ImageNetDataset
from tqdm import tqdm
import torch.optim as optim
from Model import NetModel
from torchvision.transforms import transforms
import torch.nn.functional as F
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
 Function to retrieve all training parameters using a parser
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training configuration parser")

    # Directories
    parser.add_argument('--train_dir', type=str, default='Dataset/SPLITTED/Train', help="Path to the training dataset directory")
    parser.add_argument('--val_dir', type=str, default='Dataset/SPLITTED/Test', help="Path to the validation dataset directory")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=96, help="Batch size for training")
    parser.add_argument('--num_workers', type=int, default=12, help="Number of workers for data loading")
    parser.add_argument('--base_lr', type=float, default=0.05, help="Base learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--val_step', type=int, default=1, help="Validation step frequency")

    # KNN parameters
    parser.add_argument('--knn_k', type=int, default=200, help="Number of nearest neighbors for KNN evaluation")
    parser.add_argument('--knn_t', type=float, default=0.1, help="Temperature parameter for KNN evaluation")

    # Model parameters
    parser.add_argument('--dim', type=int, default=512, help="Dimensionality of the projector's feature representation")
    parser.add_argument('--predictor_dim', type=int, default=128, help="Dimensionality of the predictor's feature representation")

    # Loss type parameters
    parser.add_argument('--stop_grad', type=bool, default=True, help="Whether to stop gradient backpropagation")
    parser.add_argument('--type_loss', type=str, default="Cosine Similarity", choices=["Cosine Similarity", "Cross Entropy Similarity"], help="Type of loss function to use")
    parser.add_argument('--symmetric', type=bool, default=True, help="Whether to use a symmetric loss")

    return parser.parse_args()

"""
 Function to perform a test on the validation set during training using a KNN, feature extraction of the dataset using only the output of the backbone
"""
def knn_validation(model, train_loader_ev, val_loader, knn_k, knn_t):
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

"""
Function to compute the predicted label using a KNN.

# 1. Compute the similarity matrix between the query features and the feature bank using matrix multiplication.
# 2. Select the top-k nearest neighbors (similarity scores and indices).
# 3. Retrieve the labels of the top-k neighbors and scale the similarity weights using the temperature parameter.
# 4. Create one-hot encodings for the labels of the top-k neighbors.
# 5. Compute the prediction scores by summing the weighted one-hot encodings for each class.
# 6. Return the predicted labels, sorted by descending prediction score.
"""

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k=200, knn_t=0.1):
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    return pred_labels


"""
Function to schedule the learning rate during training, scheduling only the model parameters labeled with fixed = False
"""
def lr_scheduler(opt, init_lr, actual_epoch, max_epoch):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * actual_epoch / max_epoch))
    for param_group in opt.param_groups:
        if 'fixed' in param_group and param_group['fixed']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

"""
Function to train the SimSiam model, also performing calls to knn_validation() for validation at each val_step. 
Additionally, it computes the mean standard deviation of the model outputs at each val_step.

The args.symmetric parameter is used to determine whether to use the symmetric version of the loss function (d1 + d2) 
or the asymmetric version (only d1).
"""
def train(model, optimizer, train_loader, train_loader_ev, val_loader, lr, args, exp):
    scaler = torch.amp.GradScaler()
    num_batches = len(train_loader)
    for epoch in tqdm(range(args.epochs), desc="Training"):
        lr_scheduler(optimizer, lr, epoch, args.epochs)
        model.train()
        running_loss = 0
        all_normalized_outputs = []
        for images_aug1, images_aug2 in train_loader:
            images_aug1 = images_aug1.to(device)
            images_aug2 = images_aug2.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type):
                d1, d2, z1, z2 = model(images_aug1, images_aug2)
                if args.symmetric:
                    loss = d1 + d2
                else:
                    loss = d1
            scaler.scale(loss).backward()  # loss.backward()
            scaler.step(optimizer)  # optimizer.step()
            scaler.update()
            running_loss += loss.item()

            if epoch % args.val_step == 0:
                with torch.no_grad():
                    z1_norm = z1 / z1.norm(dim=1, keepdim=True)
                    all_normalized_outputs.append(z1_norm)
                    z2_norm = z2 / z2.norm(dim=1, keepdim=True)
                    all_normalized_outputs.append(z2_norm)

        epoch_loss = running_loss / num_batches
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
        exp.log_metric('loss', epoch_loss, step=epoch)

        if epoch % args.val_step == 0:
            all_normalized_outputs = torch.cat(all_normalized_outputs, dim=0)
            std_per_channel = all_normalized_outputs.std(dim=0)
            avg_epoch_std = std_per_channel.mean().item()

            top1_accuracy, top5_accuracy = knn_validation(model, train_loader_ev, val_loader, args.knn_k, args.knn_t)

            print(f"Epoch: {epoch}, avg_std: {avg_epoch_std}, Top-1 Knn Accuracy: {top1_accuracy:.4f}, Top-5 Knn Accuracy: {top5_accuracy:.4f}")
            exp.log_metric('avg_std', avg_epoch_std, step=epoch)
            exp.log_metric('val_top1_accuracy', top1_accuracy, step=epoch)
            exp.log_metric('val_top5_accuracy', top5_accuracy, step=epoch)

        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                       "Models/SimSiam/model_" + str(args.epochs) + "_" + str(args.batch_size) + "_Checkpoint_" + str(epoch) + ".pth")

"""
Main function that sets up all the necessary dataloaders for training, retrieves the various parameters from the parser, 
and also defines the optimizer to be used. 
Creates and defines the model parameters. The args.stop_grad parameter is used to decide whether to apply a stop gradient on z during loss computation. 
The args.type_loss parameter is used to select the type of loss to be used during training.
"""
def main():

    comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")
    args = parse_arguments()
    lr = (args.base_lr * args.batch_size) / 256

    exp = comet_ml.Experiment(project_name="Deep Learning Project", auto_metric_logging=False, auto_param_logging=False)
    parameters = {'batch_size': args.batch_size, 'learning_rate': lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}
    exp.log_parameters(parameters)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_aug = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageNetDataset(root_dir=args.train_dir, mode="train", transform=transform_aug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    train_dataset_mode_ev = ImageNetDataset(root_dir=args.train_dir, mode="eval", transform=transform)
    train_loader_ev = DataLoader(train_dataset_mode_ev, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    val_dataset = ImageNetDataset(root_dir=args.val_dir, mode="eval", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = NetModel(dim=args.dim, predictor_dim=args.predictor_dim, stop_grad=args.stop_grad, type_loss=args.type_loss)
    model.to(device)

    optim_params = [{'params': model.backbone.parameters(), 'fixed': False},
                    {'params': model.projector.parameters(), 'fixed': False},
                    {'params': model.predictor.parameters(), 'fixed': True}]

    optimizer = optim.SGD(optim_params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train(model, optimizer, train_loader, train_loader_ev, val_loader, lr, args, exp)

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               "Models/SimSiam/model_" + str(args.epochs) + "_" + str(args.batch_size) + "_Final.pth")


if __name__ == "__main__":
    main()
