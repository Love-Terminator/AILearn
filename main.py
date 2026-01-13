from Transformer.transformer import *

'''
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
'''

if __name__ == '__main__':
    # 超参数
    model_dim = 512
    n_heads = 8
    embd_dim = 512
    maxSeqLen = 100
    encoderLayerNum = 6
    decodeLayerNum = 6
    srcVocabSize = 5000
    desVocabSize = 5000
    dropout = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transformer = Transformer(model_dim, n_heads, embd_dim, maxSeqLen, encoderLayerNum, decodeLayerNum, srcVocabSize,
                              desVocabSize, dropout).to(device)
    # 生成随机数据
    src_data = torch.randint(1, srcVocabSize, (64, maxSeqLen)).to(device) # 源序列
    tgt_data = torch.randint(1, desVocabSize, (64, maxSeqLen)).to(device)  # 目标序列

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)  # 忽略填充部分的损失
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # 训练循环
    transformer.train()
    for epoch in range(100):
        optimizer.zero_grad()  # 清空梯度，防止累积

        # 输入目标序列时去掉最后一个词（用于预测下一个词）
        output = transformer(src_data, tgt_data)

        # 计算损失时，目标序列从第二个词开始（即预测下一个词）
        # output形状: (batch_size, seq_length-1, tgt_vocab_size)
        # 目标形状: (batch_size, seq_length-1)
        loss = criterion(
            output.contiguous().view(-1, desVocabSize),
            tgt_data.contiguous().view(-1)
        )

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
