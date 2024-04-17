import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module): # 定义一个位置编码器
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__() # 调用父类 nn.Module 的初始化函数
        self.dropout = nn.Dropout(p=dropout) # 定义丢弃率
        pe = torch.zeros(max_len, d_model) # 创建全零向量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 创建位置序列
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 计算频率项
        pe[:, 0::2] = torch.sin(position * div_term) # 计算正弦位置编码
        pe[:, 1::2] = torch.cos(position * div_term) # 计算余弦位置编码
        pe = pe.unsqueeze(0) # 增加一个维度
        self.register_buffer('pe', pe)  # 创建缓冲区

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach() # 将输入与位置编码相加
        return self.dropout(x)  # 应用丢弃层
    
"""
这段代码是用于实现位置编码（Positional Encoding）的，位置编码是一种在自然语言处理（NLP）任务中应用的编码技术，尤其在 Transformer 算法中的应用较为广泛。
其目的是向模型提供单词在句子中的位置信息。由于 transformer 等模型的输入是无序的，所以需要通过这种方式提供位置信息。
位置编码采用一对正弦函数和余弦函数来为每个位置生成一个独特的编码，通过这种方式，模型就能够区分出序列中不同位置的单词。
"""