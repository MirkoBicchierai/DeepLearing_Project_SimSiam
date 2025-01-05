import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class D(nn.Module):
    def __init__(self, stop_grad, type_loss):
        super(D, self).__init__()
        self.stop_grad = stop_grad
        self.type = type_loss

    def forward(self, p, z):

        if self.stop_grad:
            z = z.detach()

        if self.type == "Cosine Similarity":
            p = F.normalize(p, p=2, dim=1)
            z = F.normalize(z, p=2, dim=1)
            return -(p * z).sum(dim=1).mean()

        if self.type == "Cross Entropy Similarity":
            z_softmax = F.softmax(z, dim=1)
            log_p_softmax = F.log_softmax(p, dim=1)
            return -(z_softmax * log_p_softmax).sum(dim=1).mean()


class NetModel(nn.Module):
    def __init__(self, dim, predictor_dim, stop_grad, type_loss):
        super(NetModel, self).__init__()

        self.d = D(stop_grad, type_loss)

        resnet18 = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
        prev_dim = resnet18.fc.in_features

        self.projector = nn.Sequential(
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)
        )

        self.predictor = nn.Sequential(nn.Linear(dim, predictor_dim, bias=False),
                                        nn.BatchNorm1d(predictor_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(predictor_dim, dim))


    def forward(self, aug1, aug2):

        out1 = self.backbone(aug1).squeeze()
        z1 = self.projector(out1)

        out2 = self.backbone(aug2).squeeze()
        z2 = self.projector(out2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return self.d(p1, z2) / 2., self.d(p2, z1) / 2., z1, z2


    def get_backbone_out(self, x):
        backbone_out = self.backbone(x).squeeze()
        return backbone_out


class LinearEvaluationModel(nn.Module):
    def __init__(self, input_dim, num_classes, backbone):
        super(LinearEvaluationModel, self).__init__()
        self.backbone = backbone
        for param in backbone.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(input_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        out = self.backbone(x)
        out = out.squeeze()
        x = self.linear(out)
        return x
