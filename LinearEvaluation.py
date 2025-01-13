import comet_ml
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from DataLoader import ImageNetDataset
from Model import LinearEvaluationModel, NetModel
import torchvision.transforms as T
import argparse
from torchinfo import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
 Function that performs a test using the dataloader passed as an argument (test_loader), 
 calculating the top-1 accuracy and top-5 accuracy.
"""

def test(model, test_loader, criterion):
    model.eval()
    total_top1, total_top5, total_num = 0, 0, 0
    total_val_loss = 0

    with torch.no_grad():
        for images, target in test_loader:
            images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, target)

            total_val_loss += loss.item() * images.size(0)

            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_top1 = correct[0].float().sum().item()
            correct_top5 = correct[:5].float().sum().item()
            total_top1 += correct_top1
            total_top5 += correct_top5
            total_num += images.size(0)

    return (total_top1 / total_num), (total_top5 / total_num), total_val_loss / total_num

"""
 Function that loads the selected dataset for linear evaluation and performs the split into training, validation, and test sets.
"""
def get_dataloader(dataset, batch_size, num_workers):
    if dataset == "MiniImageNet":
        num_classes = 100
        train_transform = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        test_transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        test_set = ImageNetDataset(root_dir="./Dataset/Split/Test/", mode="eval", transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        train_set = ImageNetDataset(root_dir="./Dataset/Split/Train/", mode="eval", transform=train_transform)
        dataset_length = len(train_set)
        train_size = int(0.8 * dataset_length)
        val_size = dataset_length - train_size

        train_subset, val_subset = random_split(train_set, [train_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if dataset == "Cifar10":
        num_classes = 10
        transform_train = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            ])

        transform_test = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            ])

        train_set = torchvision.datasets.CIFAR10(root="./Dataset/Cifar", train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root="./Dataset/Cifar", train=False, download=True, transform=transform_test)

        train_set, val_set = random_split(train_set, [40000, 10000])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)

    return test_loader, val_loader, train_loader, num_classes


"""
 Function to retrieve all training parameters using a parser
"""

def get_args():

    parser = argparse.ArgumentParser(description="Parser for Linear Evaluation training parameters")

    parser.add_argument('--key', type=str, default='S8bPmX5TXBAi6879L55Qp3eWW', help="Comet ML Api Key")
    parser.add_argument("--path", type=str, default="Models/SimSiam/Symmetric Loss/model_200_96_Final.pth",
                        help="Path to the model file")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training, default 128")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training, default 100")
    parser.add_argument("--lr", type=float, default=0.3,
                        help="Learning rate for optimizer (Used: 0.025 Cifar10, 0.3 MiniImageNet)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum value for optimizer, default 0.9")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay, default 0")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Number of workers for data loading, default 12")
    parser.add_argument("--dataset", type=str, default="MiniImageNet", choices=["Cifar10", "MiniImageNet"],
                        help="Dataset name")
    parser.add_argument("--projector_dim", type=int, default=512,
                        help="Projector dimension of pretrained model, default 512")
    parser.add_argument("--predictor_dim", type=int, default=128,
                        help="Predictor dimension of pretrained model, default 128")

    return parser.parse_args()

"""
 Function to train the LinearEvaluationModel, performing a test on the validation set at each epoch.
"""
def train(model, epochs, optimizer, scheduler, criterion, train_loader, val_loader, dataset, exp):
    model.eval()
    for epoch in tqdm(range(epochs)):
        total_loss, total_num = 0, 0
        for image, target in train_loader:
            image = image.cuda()
            target = target.cuda()
            output = model(image)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_num += train_loader.batch_size
            total_loss += loss.item() * train_loader.batch_size

        scheduler.step()

        train_loss = total_loss / total_num
        print("Epoch: ", str(epoch), "Train Loss:", str(train_loss))
        exp.log_metric(dataset + ' Linear Evaluation Loss Train', train_loss, step=epoch)

        top1, top5, val_loss = test(model, val_loader, criterion)

        print("Epoch: ", str(epoch), "Top1:", str(top1), "Top5:", str(top5), "Validation Loss:", str(val_loss))
        exp.log_metric(dataset + ' Linear Evaluation Loss Evaluation', val_loss, step=epoch)
        exp.log_metric(dataset + ' Linear Evaluation Top1 Accuracy - Validation', top1, step=epoch)
        exp.log_metric(dataset + ' Linear Evaluation Top5 Accuracy - Validation', top5, step=epoch)


"""
 Main function that retrieves all parameters from the parser, loads the pretrained model, 
 and defines the LinearEvaluationModel to be used. Additionally, it loads one of the two selected datasets: MiniImageNet or Cifar10.
 Also defines the loss used for training, specifically the CrossEntropyLoss.
"""
def main():

    args = get_args()

    comet_ml.login(api_key=args.key)
    exp = comet_ml.Experiment(project_name="Deep Learning Project", auto_metric_logging=False, auto_param_logging=False)
    parameters = {'batch_size': args.batch_size, 'learning_rate': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay, 'pretrain_model':args.path}
    exp.log_parameters(parameters)

    test_loader, val_loader, train_loader, num_classes = get_dataloader(args.dataset, args.batch_size, args.num_workers)

    checkpoint = torch.load(args.path, map_location="cuda")
    pretrained_model = NetModel(dim=args.projector_dim, predictor_dim=args.predictor_dim,stop_grad=True, type_loss="Cosine Similarity").cuda()
    pretrained_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model = LinearEvaluationModel(input_dim=args.projector_dim, num_classes=num_classes, backbone=pretrained_model.backbone).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    model = model.cuda()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    input, _ = next(iter(train_loader))
    summary(model.cuda(),input_data=input.cuda(),col_names=["input_size", "output_size", "num_params"],depth=5,)

    train(model, args.epochs, optimizer, scheduler, criterion, train_loader, val_loader, args.dataset, exp)

    top1, top5, _ = test(model, test_loader, criterion)
    exp.log_metric(args.dataset + ' Linear Evaluation Top1 Accuracy - Test', top1)
    exp.log_metric(args.dataset + ' Linear Evaluation Top5 Accuracy - Test', top5)
    print("Accuracy Test - ", "Top1:", str(top1), "Top5:", str(top5))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               "Models/LinearEval/model_" + str(args.epochs) + "_" + str(args.batch_size) + "_" + args.dataset + ".pth")


if __name__ == "__main__":
    main()