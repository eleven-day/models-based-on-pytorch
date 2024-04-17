import torch.nn as nn
import torch
import numpy as np
from DecoderLayer import DecoderLayer
from Feed_Forward import get_attn_subsequence_mask, get_attn_pad_mask

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_layers, tgt_len):
        # 初始化解码器
        super(Decoder, self).__init__()
        # 目标嵌入层
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        # 位置嵌入层，使用预先训练好的Sin/Cos编码表进行嵌入，参数freeze=True表示嵌入层参数不进行学习
        self.pos_emb = nn.Embedding.from_pretrained(self.get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        # 定义解码器层，数量为n_layers
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # 进行词嵌入和位置嵌入的相加操作
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        # 获取解码器自注意力的padding mask
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # 获取解码器自注意力的序列 mask
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(dec_inputs)
        # 将上述两个 mask 相加得到最终的解码器自注意力 mask
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # 获取解码器和编码器之间的注意力 mask
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        # 依次经过每一层解码器层的处理
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # 返回最后的输出、自注意力结果和编码器-解码器注意力结果
        return dec_outputs, dec_self_attns, dec_enc_attns

    def get_sinusoid_encoding_table(self, n_position, d_model):
        # 计算位置角
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
        # 获取位置角向量
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_model)]

        # 计算Sin和Cos的位置编码表
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)
    
"""
这段代码实现了Transformer模型的解码器部分。其主要功能是接收编码器的输出和目标序列，通过多层解码器层完成信息解码，输出最后的结果。
在每一层中，都会有两种注意力机制：自注意力机制处理解码器自身的信息，并通过注意力mask做到解码时的自回归性质；
编码器-解码器注意力机制则处理编码器的输出和解码器的信息，并通过相应的mask保证了注意力计算的正确性。
其中的位置嵌入则通过一个预先计算好的Sin/Cos函数的位置编码表完成，这使得模型可以获得序列中的位置信息。
"""