import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

# 定义一个Transformer类继承torch.nn.Module
class Transformer(nn.Module):
    # 定义Transformer的初始化函数，d_model表示模型hidden layer的维度， tgt_vocab_size表示目标词汇表的大小
    def __init__(self, d_model, tgt_vocab_size):
        # 初始化父类
        super(Transformer, self).__init__()
        # 创建编码器对象
        self.encoder = Encoder()
        # 创建解码器对象
        self.decoder = Decoder()
        # 创建线性变换层，输出层，进行词汇预测，线性变换从 d_model 维度到目标词汇表大小的维度
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    # Transformer前向传播
    def forward(self, enc_inputs, dec_inputs):
        # 算出编码器的输出以及它的self attention
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # 算出解码器的输出，它的self attention以及对编码器输出的attention
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # 得到解码器的输出
        dec_logits = self.projection(dec_outputs)
        # 返回所有结果，包含解码器最后的输出、编码器self attention，解码self attention和解码对编码器的attention
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
    
"""
这段代码的核心功能是定义和实现了一个Transformer模型，
这个模型主要包含了一个编码器(Encoder)、一个解码器(Decoder)以及一个输出层，这个输出层是用来进行词汇预测的。
在模型的前向传播中，会先通过编码器对输入进行编码，并输出编码结果和自注意力图(enc_self_attns)，
再通过解码器对编码结果进行解码，并输出解码结果、解码器的自注意力图(dec_self_attns)以及解码器对编码结果的注意力图(dec_enc_attns)。
最后，将解码结果通过输出层进行词汇预测。
"""