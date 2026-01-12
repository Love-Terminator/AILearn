from torch import nn
from Transformer.fnn import *
from Transformer.attention import *

class EncoderLayer(nn.modules):
    def __int__(self, args):
        super().__int__()

        # Attention前的归一化处理
        self.attention_norm = LayerNorm(args.embd_dim)
        # 多头注意力
        self.attention = MultiHeadAttention(args, is_causal=False)
        # fnn前的归一化处理
        self.fnn_norm = LayerNorm(args.dim)
        # 前馈神经网络
        self.fnn = FNN(args.dim, args.dim, args.dropout)

    def forward(self, x):
        norm_x = self.attention_norm.forward(x)
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        out = h + self.fnn.forward(self.fnn_norm.forward(h))
        return out

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        "分别通过 N 层 Encoder Layer"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
