import torch.nn as nn
from Multi_Head_Attention import MultiHeadAttention
from Feed_Forward import PoswiseFeedForwardNet

class EncoderLayer(nn.Module):  # 定义一个编码层类，继承自nn.Module
    def __init__(self):
        super(EncoderLayer, self).__init__()  # 调用父类的初始化方法
        self.enc_self_attn = MultiHeadAttention()  # 创建多头注意力对象，用于进行自注意力操作
        self.pos_ffn = PoswiseFeedForwardNet()  # 创建前馈神经网络对象，用于处理注意力操作后的数据

    def forward(self, enc_inputs, enc_self_attn_mask):  
        # 前向传播方法。输入参数为编码输入和自注意力掩码。
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,    
                                               enc_self_attn_mask)  
        # 调用enc_self_attn的前向传播方法，进行自注意力操作。返回编码输出和注意力权重。
        enc_outputs = self.pos_ffn(enc_outputs)  
        # 使用前馈神经网络处理自注意力操作后的结果，返回编码输出。
        return enc_outputs, attn  # 返回编码输出和注意力权重
    
"""
这段代码定义了一个Transformer模型中的编码层类 EncoderLayer，
该类包括一个多头自注意力子层（MultiHeadAttention）和一个位置前馈网络子层（PoswiseFeedForwardNet）。
输入数据首先会通过多头注意力子层进行自注意力操作，再通过位置前馈网络进行处理，最后输出处理结果。
"""