from torch import nn
from Transformer.fnn import *
from Transformer.attention import *

'''编码器'''
class EncoderLayer(nn.Module):
    def __init__(self, model_dim, embd_dim, n_heads, maxSeqLen, dropout):
        super().__init__()

        # Attention前的归一化处理
        self.attention_norm = LayerNorm(embd_dim)
        # 多头注意力
        self.attention = MultiHeadAttention(model_dim, n_heads, embd_dim, maxSeqLen, dropout, is_causal=False)
        # fnn前的归一化处理
        self.fnn_norm = LayerNorm(model_dim)
        # 前馈神经网络
        self.fnn = FNN(model_dim, embd_dim, dropout)

    def forward(self, x):
        norm_x = self.attention_norm.forward(x)
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        out = h + self.fnn.forward(self.fnn_norm.forward(h))
        return out

class Encoder(nn.Module):
    def __init__(self, model_dim, embd_dim, n_heads, maxSeqLen, dropout, encoderLayerNum):
        super(Encoder, self).__init__()
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(model_dim, embd_dim, n_heads, maxSeqLen, dropout) for _ in range(encoderLayerNum)])
        self.norm = LayerNorm(embd_dim)

    def forward(self, x):
        "分别通过 N 层 Encoder Layer"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
