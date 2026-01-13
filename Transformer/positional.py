import math

import torch
from torch import nn

#位置信息编码
class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()

        # maxSeqLen是序列最大长度
        pe = torch.zeros(args.maxSeqLen, args.embd_dim)
        # 生成位置序列
        position = torch.arange(0, args.maxSeqLen).unsqueeze(1)
        # 生成exp(log1/10000^(2i/args.maxSeqLen))
        div_term = torch.exp(
            torch.arange(0, args.embd_dim, 2) * (-math.log(10000.0)/args.embd_dim)
        )

        # 计算sin、cos的值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上,requires_grad_(False)表示不参与梯度计算
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x