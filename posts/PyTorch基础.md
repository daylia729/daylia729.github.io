---
title: Pytorch基础
description: Pytorch基础
date: 2025-08-27
tags:
  - 深度学习
---

### 环境配置
* 下载anaconda
* 创建虚拟环境
* install torch
* 下载PyCharm
* 安装ipykernel：`conda install ipykernel`
* 将环境写入Notebook的kernel中：
`python -m ipykernel install --user --name 环境名称 --display-name "Python `(环境名称)"

 
### Dataset类 加载数据集
```
import os
from torch.utils.data import Dataset
from  PIL import Image

class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir

        return img,label

    def __len__(self):
        return len(self.img_path)


root_dir = "hymenoptera/hymenoptera_data/train_data"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)

train_dataset = ants_dataset + bees_dataset
``` 
### Tensorboard的使用

```
pip install tensorboard
```
```
 tensorboard --logdir=logs
 ```

```
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

img_path = "hymenoptera/hymenoptera_data/train_data/bees/16838648_415acd9e3f.jpg"
img = Image.open(img_path)
img_array = np.array(img)

writer.add_image("test",img_array,1,dataformats='HWC')

# for i in range(100):
#     writer.add_scalar("y=x*x",i*i,i)

writer.close()
```


<img src="/public/tensorboard.png">



