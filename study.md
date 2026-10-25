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
>shape()返回形状
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
tensor([12, 15, 18, 21]) dim=0
tensor([ 6, 22, 38]) dim=1
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

