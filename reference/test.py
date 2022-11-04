from models import ConformerModel
import torch
from conformer import ConformerBlock

# t = torch.nn.Conv1d(512, 512, kernel_size=31, padding=(), groups=512)

block = ConformerBlock(dim=144, heads=4)

print(sum(p.numel() for p in block.parameters() if p.requires_grad))

model = ConformerModel(dim=144)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

x = torch.randn(1, 1024, 144)

y = model(x)

print(y.shape)
