import torch.nn
from torch import nn, optim
from Transformer.attention import *
from Transformer.decoder import *
from Transformer.encoder import *
from Transformer.fnn import *
from Transformer.positional import *

'''
model_dim: 模型维度
n_heads: 注意力头数
embd_dim: input或output的token维度
maxSeqLen: 最大序列长度
encoderLayerNum: 编码器数量
decodeLayerNum: 解码器数量
srcVocabSize: 编码词表大小
desVocabSize：解码词表大小
dropout: 损失率
'''

class Transformer(nn.Module):
    def __init__(
            self,
            model_dim,
            n_heads,
            embd_dim,
            maxSeqLen,
            encoderLayerNum,
            decodeLayerNum,
            srcVocabSize,
            desVocabSize,
            dropout):
        super().__init__()

        self.args.model_dim = model_dim
        self.args.n_heads = n_heads
        self.args.embd_dim = embd_dim
        self.args.maxSeqLen = maxSeqLen
        self.args.encoderLayerNum = encoderLayerNum
        self.args.decodeLayerNum = decodeLayerNum
        self.args.srcVocabSize = srcVocabSize
        self.args.desVocabSize = desVocabSize
        self.args.dropout = dropout

        # 注册各个模型
        self.encoder_embedding=nn.Embedding(self.args.srcVocabSize, self.args.embd_dim)
        self.decoder_embedding=nn.Embedding(self.args.desVocabSize, self.args.embd_dim)
        self.positional_encoding=PositionalEncoding(self.args)
        self.drop=nn.Dropout(self.args.dropout)
        self.encoder=Encoder(self.args)
        self.decoder=Decoder(self.args)

        # 最后的线性层，输入是 embd_dim，输出是词表大小
        self.lm_head = nn.Linear(self.args.embd_dim, self.args.desVocabSize, bias=False)

    def forward(self, src, target):
        print("srcSize: ", src.size())
        print("targetSize: ", target.size())

        # 编码器
        src_token_emb = self.encoder_embedding(src)
        src_position_emb = self.positional_encoding(src_token_emb)
        src_drop_emb = self.drop(src_position_emb)
        enc_out = self.encoder(src_drop_emb)

        # 解码器
        tar_token_emb = self.decoder_embedding(target)
        tar_position_emb = self.positional_encoding(tar_token_emb)
        tar_drop_emb = self.drop(tar_position_emb)
        dec_out = self.decoder(tar_drop_emb, enc_out)

        # 最终输出
        output = self.lm_head(dec_out)
        return output

class TransformerModel:
    def __init__(
            self,
            model_dim,
            n_heads,
            embd_dim,
            maxSeqLen,
            encoderLayerNum,
            decodeLayerNum,
            srcVocabSize,
            desVocabSize,
            dropout):
        super().__init__()

        self.srcVocabSize = srcVocabSize
        self.desVocabSize = desVocabSize
        self.maxSeqLen = maxSeqLen

        self.transformer = Transformer(model_dim, n_heads, embd_dim, maxSeqLen, encoderLayerNum, decodeLayerNum, srcVocabSize, desVocabSize, dropout)

    def train(self):
        # 生成随机数据
        src_data = torch.randint(1, self.srcVocabSize, (64, self.maxSeqLen))  # 源序列
        tgt_data = torch.randint(1, self.desVocabSize, (64, self.maxSeqLen))  # 目标序列

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充部分的损失
        optimizer = optim.Adam(self.transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        # 训练循环
        self.transformer.train()
        for epoch in range(100):
            optimizer.zero_grad()  # 清空梯度，防止累积

            # 输入目标序列时去掉最后一个词（用于预测下一个词）
            output = self.transformer(src_data, tgt_data[:, :-1])

            # 计算损失时，目标序列从第二个词开始（即预测下一个词）
            # output形状: (batch_size, seq_length-1, tgt_vocab_size)
            # 目标形状: (batch_size, seq_length-1)
            loss = criterion(
                output.contiguous().view(-1, self.desVocabSize),
                tgt_data[:, 1:].contiguous().view(-1)
            )

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")


