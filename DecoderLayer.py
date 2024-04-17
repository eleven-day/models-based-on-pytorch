import torch.nn as nn
from Multi_Head_Attention import MultiHeadAttention
from Feed_Forward import PoswiseFeedForwardNet

#定义一个解码器层的类，继承nn.Module
class DecoderLayer(nn.Module):
    def __init__(self):
        #使用super方法，调用父类的初始化方法
        super(DecoderLayer, self).__init__()
        #创建一个多头自注意力对象
        self.dec_self_attn = MultiHeadAttention()
        #创建一个多头编码-解码注意力对象
        self.dec_enc_attn = MultiHeadAttention()
        #创建一个位置前馈神经网络对象
        self.pos_ffn = PoswiseFeedForwardNet()

    #定义前向传播方法
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):                                                                                      
        #先执行自注意力操作：输入为解码器输入、解码器输入和解码器输入以及对自己的注意力掩码
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, 
                                                 dec_inputs, dec_self_attn_mask)   
                                                                                   
        #然后执行编码-解码注意力操作：输入为自注意力操作的输出、编码器的输出和编码器的输出以及对编码器的注意力掩码
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, 
                                                enc_outputs, dec_enc_attn_mask)   
                                                                                   
        #然后通过一个位置前馈神经网络
        dec_outputs = self.pos_ffn(dec_outputs)                                    
        #返回解码器输出、解码器自注意力和解码器编码-解码注意力
        return dec_outputs, dec_self_attn, dec_enc_attn
    
"""
这段代码定义了一个解码器层的类，其中使用了多头自注意力（MultiHeadAttention）和位置前馈神经网络（PoswiseFeedForwardNet）。
在前向传播方法中，先经过多头自注意力操作，将解码器的输入进行自注意力计算，然后进行编码-解码注意力计算，最后通过位置前馈网络操作。
主要用于神经网络模型Transformer中的解码器部分。
"""