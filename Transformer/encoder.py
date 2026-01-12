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