---
title: Pytorch基础
description: Pytorch基础
date: 2025-08-27
tags:
  - Pytorch
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


### Transforms的使用

<img src="/public/pytorch_learn1.png">

```
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os
os.makedirs("logs",exist_ok=True)
img_path = "hymenoptera/hymenoptera_data/train_data/bees/39747887_42df2855ee.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
# ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("Tensor_img",tensor_img)
#Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("norm_img",img_norm)


writer.close()
```


```
tensor(0.6627)
tensor(0.3255)
```
(0.6627-0.5) / 0.5 = 0.3255


<img src="/public//pytorch_learn2.png">

这些工具在图像数据的预处理增强以及准备输入到神经网络的过程中发挥着重要作用，能够帮助提升模型的训练效果以及泛化能力


### torchvision里的数据集
https://docs.pytorch.org/vision/stable/datasets.html

* 如何使用torchvision提供的标准数据集

下载慢：复制下载链接到迅雷下载

```
import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


train_dataset = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)
```

### DataLoader

https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

```
import torchvision
from torch.utils.data import DataLoader


test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

img,target = test_data[0]
print(img.shape)
print(target)

step = 0
for data in test_loader:
    imgs,targets = data
    print(imgs.shape)
    print(targets)
```
```
torch.Size([3, 32, 32])
3
torch.Size([64, 3, 32, 32])
tensor([2, 3, 5, 1, 4, 9, 1, 1, 1, 9, 0, 8, 9, 9, 1, 9, 6, 8, 9, 8, 1, 9, 5, 3,
        6, 5, 0, 7, 7, 1, 6, 4, 8, 2, 1, 8, 5, 3, 3, 5, 8, 6, 6, 2, 1, 7, 6, 6,
        5, 4, 6, 9, 9, 8, 3, 3, 0, 2, 5, 9, 7, 1, 3, 8])

```

