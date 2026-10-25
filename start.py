import torch

x=torch.arange(12,dtype=torch.float32).reshape(3,4)
y=torch.arange(12)
z=x.sum(dim=[0,1])
sy=x.mean()
print(sy)
print(x.cumsum(dim=0))