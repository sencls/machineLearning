# import torch
# import random
# from d2l import torch as d2l


# def synthetic_data(w,b,num_examples) :
#     '''生成数据'''
#     x=torch.normal(0,1,(num_examples,len(w)))
#     print(x.shape)
#     y=torch.matmul(x,w)+b
#     print(y.shape)
#     y+=torch.normal(0,0.01,y.shape)
#     return x,y.reshape((-1,1))

# true_w=torch.tensor([2,-3.4])
# true_b=4.2
# features,labels=synthetic_data(true_w,true_b,1000)
# print(features[0],labels[0])

# d2l.set_figsize()
# d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)

# def data_iter(batch_size,features,labels):
#     '''随机读取样本'''
#     num_examples =len(features)
#     indices =list(range(num_examples))
#     random.shuffle(indices) #将序列随机打乱
#     for i in range(0,num_examples,batch_size):
#         batch_indices =torch.tensor(indices[i:min(i+batch_size,num_examples)])
#         yield features[batch_indices],labels[batch_indices]
# batch_size =10

# for x,y in data_iter(batch_size,features,labels):
#     print(x,"\n",y)
#     break

# w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
# b=torch.zeros(1,requires_grad=True)

# def linreg(X,w,b):
#     '''线性回归模型'''
#     return torch.matmul(X,w)+b

# def squared_loss(y_hat,y):
#     '''均方误差'''
#     return (y_hat-y.reshape(y_hat.shape))**2/2

# def sgd(params,lr,batch_size):
#     '''小批量随机梯度下降'''
#     with torch.no_grad():
#         for param in params:
#             param-=lr* param.grad/batch_size
#             param.grad.zero_()

# lr =0.01
# num_epochs=10
# net =linreg
# loss =squared_loss

# for epoch in range(num_epochs):
#     for x,y in data_iter(batch_size,features,labels):
#         l=loss(net(x,w,b),y)
#         l.sum().backward()
#         sgd([w,b],lr,batch_size)
#     with torch.no_grad():
#         train_l=loss(net(features,w,b),labels)
#         print(f"epoch {epoch+1},loss{float(train_l.mean())}")

# print(f"w的估计误差: {true_w-w.reshape(true_w.shape)}")
# print(f'b的估计误差:{true_b-b}')

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(true_w,true_b,1000)

def load_array(date_arrays,batch_sizes,is_train=True):
    '''构造torch数据迭代器'''
    dataset=data.TensorDataset(*date_arrays)
    return data.DataLoader(dataset,batch_sizes,shuffle=is_train)

batch_size=10
data_iter=load_array((features,labels),batch_size)

print(next(iter(data_iter)))

from torch import nn
net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
loss =nn.MSELoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

num_epochs=3
for epoch in range(num_epochs):
    for x,y in data_iter:
        l=loss(net(x),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print(f"epoch {epoch+1},loss{l:f}")
