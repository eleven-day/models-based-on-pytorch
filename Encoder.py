import torch.nn as nn
from Positional_Encoding import PositionalEncoding
from EncoderLayer import EncoderLayer
from Feed_Forward import get_attn_pad_mask

# 定义一个编码器类，继承nn.Module
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_layers):
        # 使用super方法，调用父类的初始化方法
        super(Encoder, self).__init__()

        # 创建一个词嵌入对象，词汇表大小为src_vocab_size，嵌入维度为d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 创建一个位置编码对象，编码维度为d_model
        self.pos_emb = PositionalEncoding(d_model)
        # 使用nn.ModuleList创建n_layers个编码层对象
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    # 定义前向传播方法
    def forward(self, enc_inputs):
        # 将输入通过词嵌入层进行嵌入，得到嵌入输出
        enc_outputs = self.src_emb(enc_inputs)
        # 将嵌入层的输出，通过位置编码层进行编码，得到编码输出
        enc_outputs = self.pos_emb(enc_outputs)
        # 获得自注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        # 初始化一个空列表用来收集每一层的自注意力
        enc_self_attns = []
        # 遍历每一层，得到当前层的输出和自注意力
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            # 收集自注意力
            enc_self_attns.append(enc_self_attn)
        # 返回编码器的输出和所有层的自注意力
        return enc_outputs, enc_self_attns
    
"""
这段代码定义了一个编码器类，其中使用了词嵌入层（nn.Embedding）、位置编码层（PositionalEncoding）、编码层（EncoderLayer）
和获取注意力掩码的函数（get_attn_pad_mask）。在前向传播方法中，首先通过词嵌入层和位置编码层，得到编码输入的嵌入和位置编码，
然后通过自注意力掩码，随后进入多层编码层进行编码。这类编码器主要用于神经网络模型Transformer中的编码器部分。
"""