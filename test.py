import torch

x=torch.tensor([[1,0.1,2,3],[3,0,2,2.22]])
x=x.argmax(dim=1)
print(x)