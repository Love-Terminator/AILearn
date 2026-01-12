import torch
from torch import nn, Tensor

'''前馈神经网络'''
class MLP(nn.modules):
    def __int__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__int__()

        self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_dim: int):
        return self.dropout(self.l2(nn.ReLU(self.l1(input_dim))))

'''层归一化'''
class LayerNorm(nn.modules):
    def __int__(self, features: int, eps=1e-6):
        super().__int__()

        self.beta = nn.Parameter(torch.ones(features))
        self.gamma = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.beta * (x - mean)/(std + self.eps) + self.gamma