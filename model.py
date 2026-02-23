import math
import inspect '''Python自带的标准库：查看获取代码、函数、类的信息'''
from dataclasses import dataclass'''快速写一个存数据的小类，不用写一堆__init__'''
import torch'''pytorch整个深度学习框架'''
import torch.nn as nn'''pytorch的神经网络层工具，里面有nn.Linear全连接层，nn.Embedding词嵌入，nn.LayerNorm层归一化，nn.MultiheadAttenrion注意力（老版本）'''
from torch.nn import functional as F'''F是函数式API， 即直接调用函数，不用先建对象，相当于一次性工具'''
class LayerNorm(nn.Module):'''只有继承自nn.Module，这个类才能变成能被pytorch训练的层或模型'''
'''层归一化：深度学习训练技巧'''  
  '''原：def __init__(self,ndim,bias):#__init__函数是类的构造器，相当于准备零件
    super.__init__() #调用父类的构造函数
    self.weight=nn.Parameter(torch.ones(ndim))#nn.Parameter是告诉pytorch这是要训练的参数，需要更新，torch.ones（ndim）是创建一个全是一的一维张量即向量，zeros即全是零，ndim是向量长度
    self.bias=nn.Parameter(torch.zeros(ndim)) if bias else None
  def forward(self,input):#向前传播，相当于真正计算
    return F.layer_norm(input,self.weight.shape,self.weight,self.bias,1e-5)#这是pytorch自带的层归一化函数'''
 '''新：'''
def __init___(self,dim,eps=1e-5)：
'''eps是很小的数,为了防止方差是零导致除以零报错'''
super().__init__()
self,eps=eps
self.weight=nn.Parameter(torch.ones(dim))
self.bias=nn.Parameter(torch.zeros(dim))
def forward(self,input)：
return F.layer_norm(input,self.weight,self.bias,self.eps,dim)'''我写了dim，但是豆包写的是self.dim，但是又没有定义self.dim，这这个不太合适吧'''
'''只对最后一维归一化，而输入的维度dim就是最后一维（有些想不通），基本过程就是对最后一维算均值和方差，然后再规一化，也就是变成均值为零方差为一，再用weight缩放，bias偏移'''
class CausalSelfAttention(nn.Module)：
"""因果自注意力机制"""
 '''原： def __init__(self,config):#config是存放参数的小盒子
    super().__init__()
    assert config.n_embd % config.n_head==0#嵌入维度必须能被注意力头数整除
    self.c_attn=nn.Linear(config.n_embd,3*config.n_embd,bias==config.bias)#把q,k,v合成一个大线性层也就是qkv，注意力需要q查询,k键,v值三个向量
    self.c_proj=nn.Linear(config.n_embd,config.n_embd,bias==config.bias)#多头注意力计算完之后，把结果拼回去投影回原维度的线性层
    self.attn_dropout=nn.Dropout(config.dropout)
    self.resid_dropout=config.dropout
    self.dropout=config.dropout#前者是管注意力别偷懒，后者试管最终结果别死记     
    self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention')#Hasttr是检查一个东西有没有这个属性或者功能。hasttr(对象，名字），Torch.nn.functional是pytouch自带的一堆常用函数工具箱。
    if not self.flash:
       print("WARNING:using slow attention.Flash Attention requires Pytorch>=2.0")
       self.register_buffer("bias",torch.trill(torch.ones(config.block_size,config.block_size))#把一个张量存到模型里，但它不算可学习参数。这里存的就是因果掩码，也就是三角掩码作用是让模型只能看到前面的词不能偷看后面的词。torch.tril是下三角矩阵，用来做因果掩码。这里是创建一个下三角的因果掩码，把它存到模型里，名字叫bias。他不训练不更新，专门用来挡住模型偷看未来的词。使得GPT只能从左向右生成。
                                  .view(1,1,config.block_size,config.block_size ))#重新排列张量形状，把特征切成多头。
#如果能用flash就不用自己创建掩码，Flash内部会自动创建

def forward(self,x):
  B,T,C=x.size()
q,k,v=self.c_attn(x).split(self.n_embd,dim=2)
k=k.view(B,T,self.n.head,C//self.n_head).transpose(1,2)
q=q.view(B,T,self.n.head,C//self.n_head).transpose(1,2)
v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
if self.flash:
  y=torch.nn.functional.scaled_dot_product_attention
else:
att=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))#q@k是在算他们俩的相似度。Transpose是转置的意思，把矩阵的行和列互换。为什么transports里面是-2和-1，是因为在pytorch里边-1等于最后一维-2等于倒数第2维。要把k转置才能让它和Q相乘。
#这句话的意思是让q和k相乘之后数值不要太大，也不要太小，方差稳定在一防止训练崩溃。为什么乘以后面这一块可以让方差变为一，这是固定结论。K.size（-1）是最后一维的大小。
att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))#关键句：遮住未来的词。整句话的意思是把不该看到的位置直接填成负无穷。
#self.bias是因果掩码矩阵，：T是取前T个。把当前长度为T的句子里，所有未来的位置全部遮住，让模型只能看到前面不能看后面。T是现在这一批句子的实际长度不是固定数，每次输入不一样T就不一样。
att=F.softmax(att,dim=-1)#softmax是把一堆数字变成一组”加起来等于1”的概率。特点是每个数都大于等于0，加起来等于一，原来越大的数出来权重越大。
#放在注意力里是为了让分数变成权重，让模型知道看哪里。这句话是为了对最后一维做softmax
att=self.att_dropout(att)
y=att@v#这里是加权求和而不是求相似度。这里得到了一个新向量。这里是用权重提取v里的信息，用权重去乘v把信息混合起来。
y=y.transpose(1,2).contigunos().view(B,T,C)#把多头注意力的结果拼回原来的形状。transpose（1，2）是让第1维和第2维进行交换。
y=self.resid_dropout(self.c_proj(y))
#先把注意力的结果过一遍线性层进行融合，然后随机关掉一些神经元，最终得到注意力的输出。
return y
'''
'''新：'''
def __init__(self,config)：
super().__init__()
self.dim=config.n_embd
self.n_head=config.n_head
self.head_dim=self.dim//self.n_head'''每个头的维度等于总纬维度除以头的数量。'''
self.qkv=nn.Linear(self.dim,3*self.dim,bias=config.bias)
self.c_proj=nn.Linear(self.dim,self.dim,bias=config.bias)
self.resid_dropout= nn.Dropout(config.dropout)
self.register_buffer("bias",torch.tril(torch.ones(1,1,config.block_size,config.block_size)))
def forward(self,x):
  B,T,C=size()
q,k,v=self.qkv(x).split(self.n_embd,dim=2)'''dim=2是最后一维'''
k=k.view(B,T,self.n.head,C//self.n_head).transpose(1,2)'''把kqv全部切成多头分头去算注意力。transpose之后，把头放在词前面，可以让每个头都拿到一整句话'''
q=q.view(B,T,self.n.head,C//self.n_head).transpose(1,2)
v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
'''开始算注意力'''
att=(q@k.transpose(-2,-1))*(1.0/math.sqrt(self.head_dim))
att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('inf'))
att=F.softmax(att,dim=-1)
y=att@v
y=y.transpose(1,2).contiguous()
y=self.resid_dropout(self.c_proj(y))
return y'''必须要contiguous才view'''

class MLP(nn.Module):
  '''把注意力学到的关系信息，再做一轮深度加工和特征提炼。'''
'''原：  def __init__(self,config):
    super().__init__()
       self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)#把维度放大到四倍
        self.gelu    = nn.GELU()#用GELU激活
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)#缩小回去
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
'''
'''新：
#使用GELU激活之后引入了"非线性"如果没有激活两层线性层合起来就是一层，如果有了激活模型拥有了非线性表达能力能够学会上下文逻辑和语义。'''
def __init__（self，config）：
super().__init__()
self.c_fc=nn.Linear(config.n_embd,4*config.n_embd,bias=config.bias)
self.gelu=nn.GELU()
self.c_proj=nn.Linear(4*config.n_embd,config.n_embd,bias= config.bias)
self.dropout= nn.Dropout(config.dropout)
def forward(self,x):
  x=self.c_fc(x)
  x=self.gelu(x)
  x=self.c_proj(x)
  x=self.dropout(x)
  return x
  


class Block(nn.Module):
'''是gpt模型的核心功能单元，负责把输入的词向量进行一轮完整的关系建模+特征提炼，多层block可堆叠则能实现对语言的深入理解'''
    '''原：def __init__(self, config):
      #ln_1,ln_2分别归一化
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)#负责组建模词与词之间的依赖关系
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)#前馈网络层，负责对注意力后面的信息进行非线性加工

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))#注意力＋残差
        x = x + self.mlp(self.ln_2(x))
        return x'''
'''新：'''
def __init__(self.config):
  super().__init__()
  self.ln_1=nn.LayerNorm(config.n_embd,bias=config.bias)'''注意力和MLP是两个完全独立的模块，他们需要各自归一化'''
  self.att=CausalSelfAttention(config)
  self.ln_2=nn.LayerNorm(config.n_embd,bias= config.bias)
  self.mlp=MLP(config)
  def forward(self,x):
    x=x+self.att(self.ln_1(x))
    x=x+self.mlp(self.ln_2(x))
    return x


@dataclass
class GPTConfig:
    block_size: int = 1024#模型能处理的最大序列长度
    vocab_size: int = 50304 #词汇表大小，这是gpt2常用的设置。
    n_head: int = 12#注意力头的数量
    n_embd: int = 768#嵌入维度:用多少数字来表示一个词
    dropout: float = 0.0#Dropout概率这里是0.0，也就是不使用dropout，这是表示当前这个配置下不启用。前面的代码仍然保留了self.dropout层和dropout参数是为了灵活性。
    bias: bool = True 
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    def get_num_params(self, non_embedding=True)
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
      assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoint
if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] 
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
      fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fus
             

                            


   
                            


   
