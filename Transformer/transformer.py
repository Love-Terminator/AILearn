import torch.nn
from torch import nn
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
    def __init__(self, args):
        super().__init__()
        self.args = args

        # 注册各个模型
        self.encoder_embedding=nn.Embedding(args.srcVocabSize, args.embd_dim)
        self.decoder_embedding=nn.Embedding(args.desVocabSize, args.embd_dim)
        self.positional_encoding=PositionalEncoding(args)
        self.drop=nn.Dropout(args.dropout)
        self.encoder=Encoder(args)
        self.decoder=Decoder(args)

        # 最后的线性层，输入是 embd_dim，输出是词表大小
        self.lm_head = nn.Linear(args.embd_dim, args.desVocabSize, bias=False)

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

