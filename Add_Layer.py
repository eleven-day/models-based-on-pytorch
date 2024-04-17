import torch
import torch.nn as nn

# 定义一个LayerNorm类，继承自PyTorch中的nn.Module基类
class LayerNorm(nn.Module):
    # 类在初始化的时候需要d_model作为输入参数，eps默认为1e-12，
    # nn.Parameter表示此变量为需要优化的参数
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 初始化γ参数为全1的向量
        self.beta = nn.Parameter(torch.zeros(d_model))  # 初始化β参数为全0的向量
        self.eps = eps  # 为了数值稳定性（避免除以零错误）

    # 前向传播过程
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 计算平均值
        var = x.var(-1, unbiased=False, keepdim=True)  # 计算方差

        # 标准化过程：（x - 均值） / 根号（方差 + eps）
        out = (x - mean) / torch.sqrt(var + self.eps)
        # 缩放和平移：γ * 标准化数据 + β
        out = self.gamma * out + self.beta
        return out
    
"""
这段代码定义了一种计算层次标准化（Layer Normalization）的类LayerNorm。
层次标准化是一种常用的深度学习模型中的正则化技术，可以加速模型训练，提高模型的泛化能力。
这个类中，gamma和beta是该层的可学习参数，前向传播中，用输入减去其均值，再除以其标准差实现标准化，
然后乘以gamma并加上beta实现重标定，输出与输入有相同的shape，但数值已经标准化。
"""