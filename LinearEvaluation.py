import comet_ml
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoader import ImageNetDataset
from Model import LinearEvaluationModel
import torchvision.transforms as T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(optimizer, epoch, lr, schedule):
    """Decay the learning rate based on schedule"""
    lr = lr
    for milestone in schedule:
        lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def test(model, test_loader, criterion):
    model.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    total_val_loss = 0.0

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for images, target in test_bar:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            total_val_loss += loss.item() * images.size(0)

            _, pred = output.topk(5, 1, True, True)  # get top-5 (includes top-1)
            pred = pred.t()  # transpose to shape [k, batch_size]
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_top1 = correct[0].float().sum().item()  # just the first prediction
            correct_top5 = correct[:5].float().sum().item()  # any of top 5 predictions
            total_top1 += correct_top1
            total_top5 += correct_top5

            total_num += images.size(0)

    return (total_top1 / total_num), (total_top5 / total_num), total_val_loss / total_num


def main():

    path = "Models/SimSiam/model_200_96_Checkpoint_150.pth"
    batch_size = 64
    epochs = 100
    lr =0.1
    momentum=0.9
    weight_decay=0.0001
    schedule = [60,80]

    comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")
    exp = comet_ml.Experiment(project_name="Deep Learning Project", auto_metric_logging=False, auto_param_logging=False)
    parameters = {'batch_size': batch_size, 'learning_rate': lr, 'momentum': momentum, 'weight_decay': weight_decay}
    exp.log_parameters(parameters)

    pretrained_model = torch.load(path)
    pretrained_model.stop_grad = True
    pretrained_model.to(device)

    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    test_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_data = ImageNetDataset(root_dir="./Dataset/SPLITTED/Test/", mode="eval", transform=test_transform)
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True)

    train_data = ImageNetDataset(root_dir="./Dataset/SPLITTED/Train/", mode="eval", transform=train_transform)
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True)

    num_classes = train_data.num_classes

    model = LinearEvaluationModel(input_dim=512, num_classes=num_classes, backbone=pretrained_model.backbone).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr,momentum=momentum,weight_decay=weight_decay)
    model = model.cuda()

    model.eval()
    for epoch in tqdm(range(epochs)):
        adjust_learning_rate(optimizer, epoch, lr, schedule)
        total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
        for image, target in train_bar:
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_num += train_loader.batch_size
            total_loss += loss.item() * train_loader.batch_size

        train_loss = total_loss / total_num
        print("Train Loss:", str(train_loss))
        exp.log_metric('Linear Evaluation Loss Train', train_loss, step=epoch)

        top1, top5, val_loss = test(model, val_loader, criterion)
        print("Top1:", str(top1), "Top5:", str(top5), "Validation Loss:", str(val_loss))
        exp.log_metric('Linear Evaluation Loss Evaluation', val_loss, step=epoch)
        exp.log_metric('Linear Evaluation Top1 Accuracy', top1, step=epoch)
        exp.log_metric('Linear Evaluation Top5 Accuracy', top5, step=epoch)

if __name__ == "__main__":
    main()
