from models.conformer import ConformerModel
import torch

model = ConformerModel(dim=128)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

x = torch.randn(1, 1024, 128)

print(model(x).shape)