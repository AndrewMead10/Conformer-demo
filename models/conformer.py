from torch import nn
import torch.nn.functional as F


class ConformerModel(nn.Module):
    pass

class ConformerBlock(nn.Module):
    pass

class AttentionModule(nn.Module):
    pass

class FeedForwardModule(nn.Module):
    pass

class ConvModule(nn.Module):
    def __init__(self, dim, kernel_size, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.padding = self.calc_same_padding(kernel_size)

        self.layer_norm = nn.LayerNorm(dim)
        self.point1 = nn.Conv1d(dim, dim*expansion_factor*2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depth = nn.Conv1d(dim*expansion_factor, dim*expansion_factor,
                               kernel_size=kernel_size, groups=dim*expansion_factor)
        self.batch_norm = nn.BatchNorm1d(dim*expansion_factor)
        self.swish = nn.SiLU()
        self.point2 = nn.Conv1d(dim*expansion_factor, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        orig_x = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.point1(x)
        x = self.glu(x)
        x = F.pad(x, self.padding)
        x = self.depth(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.point2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return orig_x + x

    def calc_same_padding(self, kernel_size):
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)


