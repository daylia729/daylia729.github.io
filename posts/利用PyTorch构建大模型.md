---
title: 利用PyTorch构建大模型
description: 利用PyTorch构建大模型
date: 2025-09-14
tags:
  - 大模型
---

### Tensor
#### float32
#### float16
#### bfloat16
#### fp8
* https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Introduction-to-FP8

#### Implications on training
* Training with float32 works,but require lots of memory.
* Training with fp8,float16 and even bfloat16 is risky,and you can get instability.
* Solution:use mixed precision training.For example,some people like to use float32 for the attention to make sure that doesn't get messed up,for simple forward passes with matrix multiplies,bf16 is fine.

### Tensor matmul
参考之前写的数据操作那一部分

### Einops
* 是一个独立的Python库，核心作用是提供简洁直观的语法来处理张量的维度操作，且跨框架兼容
简单用法，来自https://blog.csdn.net/leviopku/article/details/116204922
```
import torch
from einops import rearrange
from einops import repeat
from einops import reduce

a = torch.randn(3,3,9)
b = rearrange(a,'h w c -> c h w')
print(b.shape)

c = rearrange(a,'c (r p) w -> c r p w',p=3)
print(c.shape)

d = torch.randn(9,9)
e = repeat(d,'h w -> c h w',c=3)
print(e.shape)

# 改变d的形状为[1,1,9,9]
d = d.unsqueeze(0).unsqueeze(0)
f = reduce(d, 'b c (h h2) (w w2) -> b h w c', 'mean', h2=3, w2=3)
print(f.shape)
# einops可以直接写在pytorch神经网络模型的layer里
# from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU
# from einops.layers.torch import Rearrange
#
# model = Sequential(
#     Conv2d(3, 6, kernel_size=5),
#     MaxPool2d(kernel_size=2),
#     Conv2d(6, 16, kernel_size=5),
#     MaxPool2d(kernel_size=2),
#     # flattening
#     Rearrange('b c h w -> b (c h w)'),
#     Linear(16 * 5 * 5, 120),
#     ReLU(),
#     Linear(120, 10),
# )
```
```
torch.Size([9, 3, 3])
torch.Size([3, 1, 3, 9])
torch.Size([3, 9, 9])
torch.Size([1, 3, 3, 1])
```