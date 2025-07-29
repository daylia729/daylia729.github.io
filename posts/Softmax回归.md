---
title: Softmax回归
description: Softmax回归
date: 2025-07-10
tags:
  - 深度学习
---

#### 回归VS分类
* 回归估计一个连续值
* 分类预测一个离散类别
#### 回归
* 单连续数值输出
* 自然区间R
* 跟真实值的区别作为损失
#### 分类
* 通常多个输出
* 输出i是预测为第i类的置信度
#### 从回归到分类-均方损失
* 对类别进行一位有效编码
$$
\mathbf{y} = \left[ y_1, y_2, \ldots, y_n \right]^T
$$
$$
y_i = 
\begin{cases}
1 & \text{if } i = y \\
0 & \text{otherwise}
\end{cases}
$$
* 使用均方损失训练
* 最大值最为预测
$$
\hat{y}=\underset{i}{\text{argmax}}o_i
$$
argmax意思是找出使oi最大时的i
#### 从回归到多类分类—无校验比例
* 对类别进行一位有效编码
* 最大值最为预测
$$
\hat{y}=\underset{i}{\text{argmax}}o_i
$$
* 需要更置信的识别正确类（大余量）
$$
o_y-o_i \geq \Delta(y,i)
$$
#### 从回归到多类分类—校验比例
* 输出匹配概率（非负，和为1）
$$
\hat{\mathbf{y}} = \text{softmax}(\mathbf{o})
$$
softmax作用于o向量，将o向量归一化，每个元素非负，和为1
$$
\hat{y}_i = \frac{\exp(o_i)}{\sum_k \exp(o_k)}
$$
(第i个类别的预测概率)
分子把原始输出映射成正数，同时也会放大差距；分母是求和，归一化，让它成为概率
* 概率 $y$ 和 $\hat{y}$ 的区别作为损失
#### Soft和交叉熵损失
* 交叉熵常用来衡量两个概率的区别
$$
H(\mathbf{p}, \mathbf{q}) = \sum_{i} - p_i \log(q_i)
$$
p是真实，q是预测，pi只有0或1；若pi=1，但是qi很小，则损失就大，模型就要优化
* 将它作为损失

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i} y_i \log \hat{y}_i = -\log \hat{y}_y
$$

* 其梯度是真实概率和预测概率的区别
推导：


<img src="/public/softmax1.jpg">


$$
\partial_{o_i} l(\mathbf{y}, \hat{\mathbf{y}}) = \text{softmax}(\mathbf{o})_i - y_i
$$

#### 损失函数 
* L2Loss

$$
l(y,y') = \frac{1}{2}(y-y')^2
$$

<img src="/public/损失函数1.jpg">

可导的光滑函数，但模型会更关注较大误差对其惩罚力度更大，对异常值敏感，影响模型训练和性能
* L1Loss

$$
l(y,y') = |y-y'|
$$

<img src="/public/损失函数2.jpg">

对异常值更鲁棒，对所有误差的处理是同等力度的，但在y-y'=0处不可导，导致训练过程不够平滑
* Huber'sRobust Loss

$$
l(y, y') = 
\begin{cases} 
\vert y - y' \vert - \frac{1}{2} & \text{if } \vert y - y' \vert > 1 \\
\frac{1}{2}(y - y')^2 & \text{otherwise}
\end{cases}
$$

<img src="/public/损失函数3.jpg">

综合了L1Loss和L2Loss的优点，但需要选择合适的阈值，增加了模型复杂度

#### 图片分类数据集
```
%matplotlib inline
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
d2l.use_svg_display()
trans = transforms.ToTensor()  
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True) 
len(mnist_train), len(mnist_test)
```
* `d2l.use_svg_display()`设置d2l用svg格式图片矢量缩放清晰
* `trans = transform.ToTensor()`将PIL图像或numpy数组转换为Pytorch张量，同时会将图像像素值归一化到[0,1]范围
* 然后加载FashionMNIST训练数据集，root是数据集存放根目录，`train=True`表示加载训练集，`download = true`则表示如果本地没有数据集则自动下载
```
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
    def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
        """绘制图像列表"""
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
           if torch.is_tensor(img):
               # 图片张量
               ax.imshow(img.numpy())
           else:
               # PIL图片
               ax.imshow(img)
               ax.axes.get_xaxis().set_visible(False)
               ax.axes.get_yaxis().set_visible(False)
        if titles:
               ax.set_title(titles[i])
        return axes
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9,titles=get_fashion_mnist_labels(y));
```
* 从训练数据集的DataLoader中获取一个批次的数据，每个批次18张图片，iter把DataLoader转换成迭代器，next获取迭代器的下一个元素（即一个批次的数据），得到特征X和标签y
* 调用`show_imges`函数展示图片，先把X调整为18张28乘28的图片，设置展示2行9列
<img src="/public/softmax2.png">

读取一小批量数据，大小为256
```
batch_size = 256

def get_dataloader_workers():  
    """使用4个进程来读取数据"""
    return 4
"""构建训练数据集的数据加载器"""
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```
* `shuffle=True`在每个epoch迭代数据前打乱数据，增加训练的随机性，有助于模型泛化
* `f'{timer.stop():.2f} sec'`遍历一轮数据加载器所花费的时间
整合所有组件，定义`load_data_fashion_mnist`函数
```
def load_data_fashion_mnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    if resize:
       trans.insert(0,transform.Resize(resize))
    trans = transforms.Compose(trans)
    minst_train = torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False,transform=tans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers()),
    data.DataLoader(mnist_test,batch_size,shuffle=True,num_workers=get_dataloader_workers()))
```
#### Softmax从零开始实现
```
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs),requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

```
* 初始化w和b，在后续反向传播可以自动计算梯度更新参数
```
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```
* 首先定义了softmax函数，可以参考上面的公式
* `torch.matmul`矩阵乘法，`reshape`函数里，-1表示自动计算，假设原X是(3,2,4),执行`X.reshape(-1,8)`第二维度指定为8，总元素数为24，则会自动计算，X的形状就变成了(3,8),这里把X的第二维度指定W的第一维度，这样才可以与W做矩阵乘法
* `cross_entropy()`是交叉熵损失函数
```
def accuracy(y_hat, y):  
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
accuracy(y_hat, y) / len(y)
```
* y的形状[batch_size],y_hat的形状是[batch_size,10]
* 先判断y_hat的形状，是否为多维张量，是否是多概率分布，如果满足条件，按行取最大值的索引，将概率分布转换为类别索引，方便和真实标签比对
* 将预测结果数据类型转换为真实标签类型，进行比对，得到一个布尔张量cmp
```
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
class Accumulator:  
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```
* 辅助类Accumulator，`d2l.size(y)`等价于`y.numel()`
```
def train_epoch_ch3(net, train_iter, loss, updater): 
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
 ```
* `isinstance`用于判断一个对象是否是某个类
```
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```
```
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```
<img src="/public/softmax3.png">

```
def predict_ch3(net, test_iter, n=6):  
    """预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

#### Softmax回归的简洁实现
通过深度学习框架的高级API能够实现softmax回归变得更加容易
```
imoprt torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
# PyTorch不会隐式地调整输入的形状。
# 因此，我们定义了展平层（flatten）在线性层前调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```
* `nn.Sequential`Pytorch中用于按顺序搭建神经网络模块的容器，会将传入的模块按顺序依次执行，前一个模块的输出作为后一个模块的输入
* `nn.Flatten`是展平层，将二维图像张量展平为一维张量，例如[batch_size,height,width,channel]变为[batch_size,height*width*channel]，方便后续全连接层处理
* `nn.Linear(784,10)是线性层，也就是全连接层，784表示输入特征的维度，10表示输出特征的维度，实现从784到10的线性变换
* 参数m表示神经网络中的模块，判断是不是线性层，是的话才执行下面的参数初始化操作
```
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```
* 但是这个train_ch3没了，所以还是把从0实现的那些函数粘过去了:smile:
