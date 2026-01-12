import math
import torch
from torch import nn, Tensor


class Transformer(nn.Module):
    def __int__(self, args):
        super().__init__()

    # 注意力计算函数
    def attention(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor, dropout=None):
        # 查询词向量维度
        d_k = query.size(-1)
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
        # 归一化处理,对每一行做归一化处理
        p_attn = scores.softmax(-1)
        # 采样，随机将某些token置0，训练模型时开启
        if dropout is not None:
            p_attn = dropout(p_attn)
        # 根据计算结果对value进行加权求和
        return torch.matmul(p_attn, value), p_attn

'''多头注意力计算'''
class MultiHeadAttention(nn.modules):
    def __int__(self, args, is_causal=False):
        super().__int__()
        # 隐藏层维度（args.dim）必须是head(args.n_heads)的整数倍
        assert args.dim % args.n_heads == 0
        # 计算每个注意力头的维度
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dim = args.dim
        # token维度
        self.embd_dim = args.embd_dim
        # dropout
        self.dropout = args.dropout

        # 参数矩阵
        self.wq = nn.Linear(self.embd_dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.embd_dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.embd_dim, self.n_heads * self.head_dim, bias=False)
        # 输出参数矩阵(dim * dim)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        # 注意力的dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        # 残差连接的dropout
        self.resid_dropout = nn.Dropout(self.dropout)
        self.is_causal = is_causal

        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        # 获取batch和序列长度
        batch, seqLen, _ = q.shape

        # 计算W、Q、V. (B, T, embd_dim) * (embd_dim, dim) --> (B, T, dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 拆分成多头. (B, T, dim) --> (B, T, n_heads, head_dim) --> (B, n_heads, T, head_dim)
        xq = xq.view(batch, seqLen, self.n_heads, self.head_dim)
        xk = xk.view(batch, seqLen, self.n_heads, self.head_dim)
        xv = xv.view(batch, seqLen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力计算. (B, n_heads, T, head_dim) * (B, n_heads, head_dim, T) --> (B, n_heads, T, T)
        scores = torch.matmul(xq, xk.transpose(-2, -1))/math.sqrt(self.head_dim)
        # 掩码自注意力
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqLen, :seqLen]
        # 计算softmax
        scores = torch.softmax(scores.float(), -1)
        # 做dropout
        scores = self.attn_dropout(scores)
        # 计算多头注意力分数. (B, n_heads, T, T) * (B, n_heads, T, head_dim) --> (B, n_heads, T, head_dim)
        output = torch.matmul(scores, xv)

        # 合并多头注意力分数. (B, n_heads, T, head_dim) --> (B, T, n_heads, head_dim) --> (B, T, dim)
        output = output.transpose(1, 2).contiguous().view(batch, seqLen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output




