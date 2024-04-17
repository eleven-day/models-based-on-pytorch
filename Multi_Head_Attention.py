import torch.nn as nn
from Self_Attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):  # 多头注意力机制类
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        # 定义相关属性
        self.n_head = n_head  # 头的数量
        self.d_k = d_k  # 键的维度
        self.d_v = d_v  # 值的维度

        # 线性变换层
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  # 用于生成查询向量的线性变换层
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)  # 用于生成键向量的线性变换层
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)  # 用于生成值向量的线性变换层
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)  # 最后的全连接层

        # 注意力机制
        self.attention = ScaledDotProductAttention(scale_factor=d_k ** 0.5)  # 缩放点积注意力

        # dropout和layer norm
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # Layer Norm层

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q  # 保留原始的查询向量

        # Layer Norm
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        # 线性变换
        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        # 交换维度准备计算注意力
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)  # 计算注意力

        # 重新排列维度，并通过全连接层
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual  # 加上残差
        q = self.layer_norm(q)  # Layer Norm
        return q, attn  # 返回结果和注意力权重
    
"""
该段代码是实现多头注意力机制（Multi-Head Attention）的代码，多头注意力机制是在自注意力模型中用于提取不同位置不同深度信息的重要部分。
每一“头”都有各自的查询、键、值权重，通过这种方式，网络可以关注输入的不同部分，这样有助于模型在理解文本的多重语义方面做得更好。
"""