---
title: Pytorch基础3
description: Pytorch基础3
date: 2025-08-31
tags:
  - Pytorch
---

### 完整的模型训练

train:
```
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor()
                                          ,download=False)

test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor()
                                         ,download=False)

train_data_size = len(train_data)
test_data_size = len(test_data)
#
# print(train_data_size)
# print(test_data_size)

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# 搭建模型

mynet = MyNet()
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(mynet.parameters(),lr=learning_rate)

total_train_step = 0
total_test_step = 0

epoch = 10

writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("第{}轮训练开始".format(i+1))

    for data in train_dataloader:
        imgs,targets = data
        outputs = mynet(imgs)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        if total_train_step % 100 == 0:
            print("训练次数：{},loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = mynet(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("total_loss:{}".format(total_test_loss))
    print("在测试集上的正确率{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    total_test_step +=1
    
    
    torch.save(mynet,"mynet_{}.pth".format(i))

writer.close()
```
model:
```
import torch
from torch import nn



class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    mynet = MyNet()


```
