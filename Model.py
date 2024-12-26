import torch.nn as nn
import torchvision.models as models


class NetModel(nn.Module):
    def __init__(self, dim, predictor_dim, stop_grad):
        super(NetModel, self).__init__()
        self.stop_grad = stop_grad
        self.encoder = models.resnet18(num_classes=dim, zero_init_residual=True, pretrained=False)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim))
        self.encoder.fc[6].bias.requires_grad = False

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, predictor_dim, bias=False),
                                        nn.BatchNorm1d(predictor_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(predictor_dim, dim))


    def forward(self, aug1, aug2):

        z1 = self.encoder(aug1)
        z2 = self.encoder(aug2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        if self.stop_grad:
            return p1, p2, z1.detach(), z2.detach()
        else:
            return p1, p2, z1, z2


class LinearEvaluationModel(nn.Module):
    def __init__(self, input_dim, output_dim, backbone):
        super(LinearEvaluationModel, self).__init__()
        self.backbone = backbone
        for param in backbone.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        _, _, z1, _ = self.backbone(x,x)
        x = self.linear(z1)
        return x
