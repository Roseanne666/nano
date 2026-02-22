import math
import inspect '''Python自带的标准库：查看获取代码、函数、类的信息'''
from dataclasses import dataclass'''快速写一个存数据的小类，不用写一堆__init__'''
import torch'''pytorch整个深度学习框架'''
import torch.nn as nn'''pytorch的神经网络层工具，里面有nn.Linear全连接层，nn.Embedding词嵌入，nn.LayerNorm层归一化，nn.MultiheadAttenrion注意力（老版本）'''
from torch.nn import functional as F'''F是函数式API， 即直接调用函数，不用先建对象，相当于一次性工具'''
class LayerNorm(nn.Module):'''只有继承自nn.Module，这个类才能变成能被pytorch训练的层或模型'''
'''层归一化：深度学习训练技巧'''  
  def __init__(self,ndim,bias):'''__init__函数是类的构造器，相当于准备零件'''
    super.__init__() '''调用父类的构造函数'''
    self.weight=nn.Parameter(torch.ones(ndim))'''nn.Parameter是告诉pytorch这是要训练的参数，需要更新，torch.ones（ndim）是创建一个全是一的一维张量即向量，zeros即全是零，ndim是向量长度'''
    self.bias=nn.Parameter(torch.zeros(ndim)) if bias else None
  def forward(self,input):'''向前传播，相当于真正计算'''
    return F.layer_norm(input,self.weight.shape,self.weight,self.bias,1e-5)
    
class CausalSelfAttention(nn.Module)：
"""因果自注意力机制"""
  def __init__(self,config):'''config是存放参数的小盒子'''
    super().__init__()
    assert config.n_embd % config.n_head==0'''嵌入维度必须能被注意力头数整除'''
    self.c_attn=nn.Linear(config.n_embd,3*config.n_embd,bias==config.bias)'''把q,k,v合成一个大线性层也就是qkv，注意力需要q查询,k键,v值三个向量'''
    self.c_proj=nn.Linear(config.n_embd,config.n_embd,bias==config.bias)'''多头注意力计算完之后，把结果拼回去投影回原维度的线性层'''
    self.attn_dropout=nn.Dropout(config.dropout)
    self.resid_dropout=config.dropout
    self.dropout=config.dropout'''前者是管注意力别偷懒，后者试管最终结果别死记'''     
    self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention')'''Hasttr是检查一个东西有没有这个属性或者功能。hasttr(对象，名字），Torch.nn.functional是pytouch自带的一堆常用函数工具箱。'''
    if not self.flash:
       print("WARNING:using slow attention.Flash Attention requires Pytorch>=2.0")
       self.register_buffer("bias",torch.trill(torch.ones(config.block_size,config.block_size))'''把一个张量存到模型里，但它不算可学习参数。这里存的就是因果掩码，也就是三角掩码作用是让模型只能看到前面的词不能偷看后面的词。torch.tril是下三角矩阵，用来做因果掩码。这里是创建一个下三角的因果掩码，把它存到模型里，名字叫bias。他不训练不更新，专门用来挡住模型偷看未来的词。使得GPT只能从左向右生成。'''
                                  .view(1,1,config.block_size,config.block_size ))'''重新排列张量形状，把特征切成多头。'''
'''如果能用flash就不用自己创建掩码，Flash内部会自动创建'''
def forward(self,x):
  B,T,C=x.size()
q,k,v
   
