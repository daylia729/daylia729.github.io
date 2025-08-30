---
title: Pytorch基础2
description: Pytorch基础2
date: 2025-08-29
tags:
  - Pytorch
---


### nn.Module的使用

https://docs.pytorch.org/docs/stable/nn.html


### 卷积操作
https://github.com/qiaohaoforever/DeepLearningFromScratch/blob/master/%E3%80%8A%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8%EF%BC%9A%E5%9F%BA%E4%BA%8EPython%E7%9A%84%E7%90%86%E8%AE%BA%E4%B8%8E%E5%AE%9E%E7%8E%B0%E3%80%8B%E9%AB%98%E6%B8%85%E4%B8%AD%E6%96%87%E7%89%88.pdf


以`covn2d`为例

先看参数：
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d
<img src="/public/pytorch基础21.png">

```
import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
output = F.conv2d(input,kernel,stride=1)
print(output)
```
```
tensor([[[[10, 12, 12],
          [18, 16, 16],
          [13,  9,  3]]]])

```


### 卷积层

<img src="/public/pytrch基础22.png">

```
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64)

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)

        return x

mynet = Mynet()

writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs,targets = data
    output = mynet(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input",imgs,step)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step = step + 1

writer.close()
```


### MaxPooling

代码把上面的`Conv2d`改成`MaxPool2d`

原理看鱼书
效果：
<img src="/public/pytorch基础23.png">

### 非线性激活

激活函数 
```
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet,self).__init__()
        self.ReLU1 = ReLU()
        self.Sigmoid1 = Sigmoid()

    def forward(self,x):
        x = self.ReLU1(x)
        # x = self.Sigmoid(x)
        return x

```
### 线性层...

看官方文档

### Sequential的使用

```
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

import torch


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10),
        )

    def forward(self,x):
        x = self.model1(x)
        return x

myNet = MyNet()
print(myNet)
x = torch.ones((64,3,32,32))
out=myNet(x)
print(out.shape)
```

```
torch.Size([64, 10])
```

<img src="/public/pytorch基础24.jpg">

### 损失函数以及反向传播

Softmax回归里讲过L1、L2（MSE）以及交叉熵损失函数

反向传播看鱼书

### 优化器Optimizer

https://docs.pytorch.org/docs/stable/optim.html

### 现有模型的修改
```
import torchvision.models
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=True)

print(vgg16)

train_data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=False)

vgg16.classifier.add_module('add_linear',nn.Linear(1000,10))

print(vgg16)
```

### 模型保存与读取
```
vgg16 = torchvision.models.vgg16(pretrained=False)
#保存方式1 模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")
# 保存方式2 模型参数(推荐)
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
```