from torch import nn
from Transformer.fnn import *
from Transformer.attention import *

'''解码器'''
class DecoderLayer(nn.Module):
    def __init__(self, model_dim, embd_dim, n_heads, maxSeqLen, dropout):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.attention_norm_1 = LayerNorm(embd_dim)
        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attention = MultiHeadAttention(model_dim, n_heads, embd_dim, maxSeqLen, dropout, is_causal=True)
        self.attention_norm_2 = LayerNorm(embd_dim)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(model_dim, n_heads, embd_dim, maxSeqLen, dropout, is_causal=False)
        self.ffn_norm = LayerNorm(embd_dim)
        # 第三个部分是 MLP
        self.feed_forward = FNN(model_dim, embd_dim, dropout)

    def forward(self, x, enc_out):
        # Layer Norm
        norm_x = self.attention_norm_1(x)
        # 掩码自注意力
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
        # 交叉注意力
        norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Decoder(nn.Module):
    def __init__(self, model_dim, embd_dim, n_heads, maxSeqLen, dropout, decoderLayerNum):
        super(Decoder, self).__init__()
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(model_dim, embd_dim, n_heads, maxSeqLen, dropout) for _ in range(decoderLayerNum)])
        self.norm = LayerNorm(embd_dim)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)

