from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
def struct():
    '''演示torch框架的编排+无限套娃'''    
    net =nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

    X=torch.rand(2,20)
    print(X)
    print(net(X))

    class MLP(nn.Module):
        def __init__(self) :
            super().__init__()
            self.hidden=nn.Linear(20,256)
            self.out=nn.Linear(256,10)

        def forward(self,x):
            return self.out(F.relu(self.hidden(x)))     

    net=MLP()
    y=torch.rand(2,20)
    y=net(y)
    print(y)  

    class MySequential(nn.Module):
        def __init__(self, *args):
            super().__init__()
            for block in args:
                self._modules[block]=block

        def forward(self,X):
            for block in self._modules.values():
                X=block(X)
            return X

    net=MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
    print(net(X))


def paramsManager():
    net=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
    x=torch.rand(size=(2,4))
    print(net(x))

    print(net[2].state_dict())

if __name__=="__main__":
    paramsManager()