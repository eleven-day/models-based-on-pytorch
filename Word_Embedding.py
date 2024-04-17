import torch.nn as nn
import math

class Embeddings(nn.Module):  
    # 初始化类
    # d_model: 嵌入向量的维数
    # vocab: 词汇表的大小
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()  # 调用父类的初始化方法
        self.lut = nn.Embedding(vocab, d_model)  # 创建一个嵌入层，vocab表示词汇表的大小，d_model表示嵌入向量的维数
        self.d_model = d_model  # 嵌入向量的维数

    # 前向传播函数
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # 返回嵌入层的输出经过缩放处理（乘以嵌入向量维数的平方根）
    
"""
这段代码中定义的类Embeddings是一个词嵌入模型, 主要用于将文本数据中的每个单词映射到一个高维的向量，以便在深度学习模型中使用。
这个类继承自PyTorch的nn.Module类，可作为深度学习模型中的一部分。

初始化函数__init__包含两个参数，分别是词嵌入向量的维数d_model和词汇表的大小vocab。
然后在初始化函数中创建一个名为self.lut的嵌入层，该层可以将输入的每个单词映射到一个d_model维的向量。

在前向传播函数forward中，输入x被送到嵌入层self.lut中，并得到嵌入向量。然后，这些嵌入向量会被缩放，即乘以嵌入向量的维数d_model的平方根。
若干个缩放操作可以帮助控制嵌入向量的大小，使得其在计算后续模型的损失函数时更加稳定。
"""