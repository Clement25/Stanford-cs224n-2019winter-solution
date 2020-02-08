import torch

x = torch.rand(3,5)
print(x,x.flatten())

y = torch.rand(4,4,5,device="cpu")
print(y.device)
print(y,y.view(16,-1))