from Transformer.transformer import *

if __name__ == '__main__':
    # 超参数
    model_dim = 512
    n_heads = 8
    embd_dim = 512
    maxSeqLen = 100
    encoderLayerNum = 6,
    decodeLayerNum = 6,
    srcVocabSize = 5000,
    desVocabSize = 5000,
    dropout = 0.1

    model = TransformerModel(
        model_dim=model_dim,
        n_heads=n_heads,
        embd_dim=embd_dim,
        maxSeqLen=maxSeqLen,
        encoderLayerNum=encoderLayerNum,
        decodeLayerNum=decodeLayerNum,
        srcVocabSize=srcVocabSize,
        desVocabSize=desVocabSize,
        dropout=dropout
    )
    model.train()
