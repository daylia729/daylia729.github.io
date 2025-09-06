---
title: RNN&LSTM
description: NLP
date: 2025-09-01
tags:
  - 深度学习
---
# RNN
【【循环神经网络】5分钟搞懂RNN，3D动画深入浅出】 https://www.bilibili.com/video/BV1z5411f7Bm/?share_source=copy_web

https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html

```
from torch import nn
import torch


# 单层单向rnn
rnn = nn.RNN(input_size=3,hidden_size=3,num_layers=1,batch_first=True)

input = torch.randn(1,2,3) # batch_size * seq_len * feature length(input_size)

output,h_n = rnn(input)
# output 是rnn每个时间步的输出，形状为batch_size * seq_len * hidden_size
# h_n是最后一步时间步的输出，形状为num_layers * batch_size * hidden_size       
print(output,h_n)
print(output.shape) #
print(h_n.shape)

torch.Size([1, 2, 3])
torch.Size([1, 1, 3])
```

* 更新隐藏层状态：
$$
h_t = \phi(W_{hh} h_{t-1} + W_{hx} x_{t-1} + b_h)
$$

* 输出：
$$
o_t = \phi(W_{oh} h_t + b_o)
$$

* $\phi$ 是激活函数

### LSTM

<img src="/public/LSTM.jpg">

* input gate
* forget gate
* output gate

### Attention机制
* 卷积、全连接、池化层都只考虑不随意线索
* 注意力机制则显示考虑随意线索，根据随意线索（查询query）来有偏向性选择输入，每个输入是一个值（value）和随意线索（key）的对

$$
f(x) = \sum_{i=1}^{n} \text{softmax}\left( -\frac{1}{2} \left( (x - x_i) w \right)^2 \right) y_i
$$
* 相对于之前的非参注意力池化层引入了w，w是可以学习的（使用高斯分布）





