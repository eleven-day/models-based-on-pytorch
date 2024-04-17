import torch.nn as nn
import torch

# 定义一个名为 ScaledDotProductAttention 的类，继承 nn.Module
class ScaledDotProductAttention(nn.Module):

    # 初始化函数，在创建一个 ScaledDotProductAttention 对象时，会自动调用这个函数
    def __init__(self, scale_factor, dropout=0.0):
        super().__init__()  # 调用父类 nn.Module 的初始化函数
        self.scale_factor = scale_factor  # 缩放因子，用于对 q 查询进行缩放
        self.dropout = nn.Dropout(dropout)  # 定义_dropout函数，用于随机忽略一些神经元，防止过拟合

    # 定义 forward 函数，这是 nn.Module 的核心，它定义了每次执行前向传播时，模块如何处理输入的数据
    def forward(self, q, k, v, mask=None):
        # 利用  q 和 k 的矩阵乘法计算注意力得分，并对其缩放
        attn = torch.matmul(q / self.scale_factor, k.transpose(2, 3))

        # 如果 mask 不为 None，则在 softmax 前对 attn 进行 mask 操作
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  
        
        # 采用 softmax 函数，并在其之后执行 dropout 操作
        attn = self.dropout(torch.softmax(attn, dim=-1))

        # 最后，通过 attn 和 v 的矩阵乘法得到最终输出
        output = torch.matmul(attn, v)
      
        return output, attn
    
"""
这段代码定义了一个名为ScaledDotProductAttention的类，这是用于处理下一层输入的注意力机制的一种常用方式——缩放点积注意力。
这个机制是Transformer模型中关键的组成部分。

这种机制的目的是决定网络的每一层在处理输入时对于不同输入元素的专注程度。
简而言之，该模型在处理输入数据时，会给不同的输入元素分配不同的“关注权重”，这就是所谓的“注意力”。

实际操作中，它通过 q（Query）和 k（Keys）的矩阵乘法来计算注意力，并在此后对结果进行缩放，然后用 softmax 函数处理这些得分。
对得分执行了 dropout 操作以预防过拟合后，将处理过的注意力与 v（Values）相乘以得到最终输出。
"""