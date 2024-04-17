import torch.nn as nn
import numpy as np
import torch

# 定义位置逐点前馈网络(PoswiseFeedForwardNet)类，继承自nn.Module
class PoswiseFeedForwardNet(nn.Module):
    # 初始化函数
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()    # 调用父类初始化函数
        self.d_model = d_model    # 将输入参数d_model赋值给self.d_model

        # 定义全连接层：包含两个线性层和一个激活函数ReLU。第一层将输入维度从d_model转为d_ff，然后ReLU作为非线性层，最后一层将维度从d_ff转回d_model
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, self.d_model, bias=False)
        )
        
        # 定义层标准化LayerNorm，修正和归一化层输入，可用于加速训练过程并提高模型稳定性 
        self.layer_norm = nn.LayerNorm(self.d_model).cuda()
        
    # 定义前向传播函数
    def forward(self, inputs):                             
        # 残差连接，将输入信息保留下来
        residual = inputs
        # 全连接层处理输入
        output = self.fc(inputs)
        # 通过层标准化返回加了残差连接的结果，即网络最终的输出
        return self.layer_norm(output + residual)

"""
这段代码定义了一个位置逐点前馈网络（Position-wise Feed Forward Network），这是 Transformer 模型的重要组成部分。
它是由两层完全连接的前馈神经网络组成，并在输入和输出之间有残差连接，并经过层标准化处理。
此网络广泛用于自然语言处理和其他相关的机器学习任务。
"""
    
def get_attn_pad_mask(seq_q, seq_k): 
    # 获取 seq_q 和 seq_k 的大小，其中 seq_q 和 seq_k 是需要进行比较的两个序列
    batch_size, len_q = seq_q.size() 
    batch_size, len_k = seq_k.size() 
    # 创建一个 mask，其值取决于 seq_k 中哪些位置的值为0
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) 
    # 扩展 mask 以匹配输入序列的长度
    return pad_attn_mask.expand(batch_size, len_q, len_k) 

def get_attn_subsequence_mask(seq): 
    # 设置mask的大小与序列的大小相同。因为我们将使用一个三角形的上三角，所以 seq.size(1) 被取了两次
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] 
    # 使用np.triu和np.ones创建一个上三角形的二维矩阵，然后转换为 torch ByteTensor
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) 
    subsequence_mask = torch.from_numpy(subsequence_mask).byte() 
    # 返回创建好的mask
    return subsequence_mask