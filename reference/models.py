import torch
from torch import nn
import torch.nn.functional as F


class ConformerModel(nn.Module):
    def __init__(self, dim=144, n_heads=4, n_blocks=1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        self.blocks = nn.Sequential(
            *[ConformerBlock(dim, n_heads) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim, n_heads, expansion_factor=4, kernel_size=31):
        super().__init__()

        self.ff1 = FeedForwardModule(dim, expansion_factor)
        print('ff1_params', sum(p.numel()
              for p in self.ff1.parameters() if p.requires_grad))
        self.att = AttentionModule(dim, n_heads)
        print('att_params', sum(p.numel()
              for p in self.att.parameters() if p.requires_grad))
        self.conv = ConvModule(dim, kernel_size)
        print('conv_params', sum(p.numel()
              for p in self.conv.parameters() if p.requires_grad))
        self.ff2 = FeedForwardModule(dim, expansion_factor)
        print('ff2_params', sum(p.numel()
              for p in self.ff2.parameters() if p.requires_grad))
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = 0.5 * self.ff1(x)
        x = self.att(x)
        x = self.conv(x)
        x = 0.5 * self.ff2(x)
        return self.layer_norm(x)


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


class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()

        self.layernorm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim*expansion_factor)
        self.swish = nn.SiLU()
        self.linear2 = nn.Linear(dim*expansion_factor, dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        orig_x = x
        x = self.layernorm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return orig_x + x


class AttentionModule(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads

        self.layer_norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, n_heads)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        orig_x = x
        x = self.layer_norm(x)
        x, _ = self.mha(x, x, x)
        x = self.dropout(x)
        return orig_x + x
