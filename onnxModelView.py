import torch

from Model import NetModel

f = NetModel(2048,512)
torch.save(f, "Models/pretrained_model.pth")
batch_size = 32
input_data = torch.randn(batch_size, 3, 224, 224)
torch.onnx.export(model=f, args=(input_data, input_data), f="test.onnx", export_params=False)