# 数据操作
N维数组是神经网络和机器学习的主要数据结构
- 0-d 标量 区分类别
- 1-d 向量 特征向量
- 2-d 矩阵 特征矩阵
- 3-d 类似于RGB图片(width&height&channel)，channel通道
- 4-d RGB图片批量(patch批量大小&width&height&channel)
- 5-d 视频批量(patch&hour&width&height&channel)

### 创建数组
- 访问元素
python的片选操作：[x,y]->x行y列
**:**表示缺省，则是直接取全部；**::num**表示隔num个取一次行或列
[x1:x2,:] 取x1到x2行,[::2,::2] 隔两行隔两列选取
[-1]返回最后的行、列,[x1:x2]左闭右开区间，可以直接指定区域进行元素修改
- 创建 
>arange(num,dtype)创建一维数组(dtype指定数据类型) 
>dim()返回维度数
>shape()返回形状,本质上是原始内存的重新分配，不涉及到新内存的分配，只有使用clone时才显式拷贝
>numel()返回元素个数
>reshape(arg1,arg2,arg3,...)不改变元素数量和值而改变张量形状
>zeros,ones形成全零，单位矩阵
>也可已通过tensor([[]])直接创建特定值
>常见运算都支持，都是直接按元素进行运算，**这是幂运算,exp()，==也是按元素运算
>cat((x,y),dim=) 将x和y在dim维合并,dim=0是最外层，依次向内
>sum 求和

**广播机制**：两个张量运算形状不一样时发生，规则：
从最右侧开始比较，两个维度相同保持不变，其中一方为1则拉伸为另一方的大小，其他情况直接报错
>numpy()转换为numpy类型，直接通过一维元素通过item()获得python的标量

### 如何读写csv
```python
import pandas as pd
import os
os.makedirs(os.path.join("../machineLearning","data"),exist_ok=True)
data_file =os.path.join('../machineLearning',"data",'house_ti.csv')
with open(data_file,'w') as f:
    f.write("num,alley,price\n")
    f.write("na,pave,1111")
data =pd.read_csv(data_file)
print(data)
```
### 处理缺失数据
一般采用丢弃或者插值
iloc选取其中的部分列，fillna(sth.mean())->将空缺值填为中指
get_dummies(sth,dummy_na=True)将类别中的NAN视作一个类别，再列一列
values()取data的值，然后用torch.tensor()转换为tensor

# 线性代数
len()可以返回张量的长度,多维度是返回dim=0的长度
.T表示转秩
.clone() 显视的分配新内存来拷贝
.sum(dim/axis) 缺省就是简单的元素全部相加，dim存在时按dim维度运算，比如dim=0，就直接将0维压缩,keepdims=True就是运算后仍保留维度，方便后续广播
```txt //疑似错误
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
tensor([18, 22, 26]) dim=0
tensor([ 3, 12, 21, 30]) dim=1
tensor(66)  dim=[0,1] 两个维度求和
```
.mean() 求均值,也可用按轴求均值dim

.cumsum 按轴求和

.dot(x,y) 点积 等价于sum(x*y)

.mv(x,y) 矩阵向量积 matrix*vector，变成一维向量

.mm(x,y) 矩阵乘法

.norm(x) x的$L_2$范数 $\lVert x \rVert_2 = \sqrt{\sum_{i=1}^{n} x_i^{2}}$

.abs(x).norm x的$L_1$范数 $\lVert x \rVert_1 = \sum_{i=1}^{n}x$

.norm(torch.ones(x,y)) F范数 $\lVert x \rVert_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^n x_{ij}^{2}}$

# 矩阵计算
亚导数：该点不可微，左导数与右导数不一致，最后该点没有导数，于是就将这个点的导数设置为[l,r]区间中的任意值
> $f(x)=\lvert x \rvert ->x=0处导数为[-1,1]$

矩阵导数：
y,x要区分各自是什么状态：标量，向量，矩阵
- 分母布局法：列向量 $\frac{\partial y}{\partial \mathbf{x}}=\begin{bmatrix}\frac{\partial y}{\partial x_1},\frac{\partial y}{\partial x_2},...,\frac{\partial y}{\partial x_n}\end{bmatrix}^{\top}$
  
- 分子布局法：行向量 $\frac{\partial y}{\partial \mathbf{x}}=\begin{bmatrix}\frac{\partial y}{\partial x_1},\frac{\partial y}{\partial x_2},...,\frac{\partial y}{\partial x_n}\end{bmatrix}$
## 结果
- 都是标量 就是正常导数
- 一个标量，一个向量 
>同上方的分子分母布局法示例
- 都是向量
>$\mathbf{x}=\begin{bmatrix}x_1,x_2,...,x_n\end{bmatrix}^{\top},\mathbf{y}=\begin{bmatrix}y_1,y_2,...,y_m\end{bmatrix}^{\top}$
>$$\frac{\partial \mathbf{y}}{\partial\mathbf{x}}=\begin{bmatrix}
    \frac{\partial y_1}{\partial \mathbf{x}},\frac{\partial y_2}{\partial \mathbf{x}},...,\frac{\partial y_m}{\partial \mathbf{x}}
\end{bmatrix}^{\top}=\begin{bmatrix}
    \frac{\partial y_1}{\partial x_1},\frac{\partial y_1}{\partial x_2},...,\frac{\partial y_1}{\partial x_n}\\
    \frac{\partial y_2}{\partial x_1},\frac{\partial y_2}{\partial x_2},...,\frac{\partial y_2}{\partial x_n}\\...\\
    \frac{\partial y_m}{\partial x_1},\frac{\partial y_m}{\partial x_2},...,\frac{\partial y_m}{\partial x_n}
\end{bmatrix}$$
- 都是矩阵
> 矩阵Y(m,l) X(n,k)->结果是(m,l,k,n)的矩阵

## 规律
分子的维度保持不变，但是分母维度要转秩然后拼接到分子维度后，就是最后结果的维度

## 自动求导
主要依赖于链式法则
向量链式法则： $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}=\frac{\partial \mathbf{y}}{\partial \boldsymbol{\mu}}\frac{\partial \boldsymbol{\mu}}{\partial \mathbf{x}}$

### instance
1. x,w是长度为n的向量，y是标量，<>是内积运算，$z=(\langle\mathbf{x},\mathbf{w}\rangle-y)^2,求\frac{\partial z}{\partial \mathbf{w}}$
>$$a=\langle\mathbf{x},\mathbf{w}\rangle,b=a-y,z=b^2\\\frac{\partial z}{\partial \mathbf{w}}=\frac{\partial z}{\partial b}\frac{\partial b}{\partial a}\frac{\partial a}{\partial \mathbf{w}}=2b*\mathbf{{x}^{\top}}$$
2. $\mathbf{X}$是矩阵，w，y是向量，$z=\lVert \mathbf{X}\mathbf{w}-\mathbf{y}\rVert_2,求\frac{\partial z}{\partial \mathbf{w}}$
>$$a=\mathbf{X}\mathbf{w},b=a-\mathbf{y},z=\lVert b\rVert^2\\frac{\partial z}{\partial \mathbf{w}}=\frac{\partial z}{\partial b}\frac{\partial b}{\partial a}\frac{\partial a}{\partial \mathbf{w}}=2b^{\top}*\mathbf{I}*\mathbf{X}$$

自动求导在于计算一个函数在指定值上的导数

### 计算图
将代码分解为op，算子，将计算表示成一个无环图，显式构造或隐式
正向累积：就是正常从input到output，反向累积也就是backforward，反向传播
计算复杂度两者在于O(n),n是算子个数，内存不一样前者是O(n),后者是O(1),正向要保留所有的中间结果

.requires_grad_(True)，或者arange(num,requires_gred=True) 存储梯度
.backward() 反向传播函数，自动计算y关于x的每个分量的梯度
.grad_zeros_() 默认情况下torch会累积梯度，更换y时，需要清除之前的值
y如果是向量，一般会将其sum()成标量，然后在backward()
detach() 在于将结果分离出去，而不是将其作为计算图的一部分

# 线性回归
## def
有n维输入 $\mathbf{X} =\begin{bmatrix}
    x_1,x_2,...x_n
\end{bmatrix}^{\top}$
线性模型有一个n维权重和标量偏差 $\mathbf{W}=\begin{bmatrix}
    w_1,w_2,...,w_n
\end{bmatrix}^{\top},b$
则输出是输入的加权和：$y=\langle\mathbf{W},\mathbf{X}\rangle+b$
可以视作单层神经网络
衡量预测质量：平方损失：$\mathbb{l}(y,\hat{y})=\frac{1}{2}(y-\hat{y})^2$
## 训练
收集训练数据，设n个样本$\mathbf{X}=\begin{bmatrix}
    \mathbf{X}_1,\mathbf{X}_2,...,\mathbf{X}_n
\end{bmatrix}^{\top}$ 
实际数值$\mathbf{y}=\begin{bmatrix}
    y_1,y_2,...,y_n
\end{bmatrix}^{\top}$

则损失函数为：$\mathbb{l}(\mathbf{X},\mathbf{y},\mathbf{w},b)=\frac{1}{2n}\sum_{i=1}^n(y_i-\langle\mathbf{x_i},\mathbf{w}\rangle-b)^2$。然后最小化loss从而学习参数w,b。存在显式解，loss只要是凸函数，就有最优解

## 基础优化方法
没有显式最优解，通过以下方式来进行解决
- 梯度下降
随机挑选一个初始值$w_0$,重复迭代参数$t=1,2,3 \mathbf{w}_t=\mathbf{w}_{t-1}-\eta\frac{\partial l}{\partial \mathbf{w}_{t-1}}$,沿梯度方向将增加loss，学习率：步长的超参数，不能太小也不能太大
- 小批量随机梯度下降
一个深度神经网络模型可能需要数分钟至数小时，计算梯度太贵，可以通过随机采样b个样本来近似损失，b是批量大小，另外一个超参数
### 基础实现
