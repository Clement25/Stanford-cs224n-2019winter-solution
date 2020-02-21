import numpy as np
import torch

""" a = np.array([[1,2,3],[4,5,6]])
print(a.shape)
b = np.array([0.5,1,2])
print(np.matmul(a,b))
print(np.dot(a,b))
print(a[[1,1],:]) """

a = torch.tensor([[2,5,3,4,0],[1,2,3,0,0]])
b = torch.sum(a,dim=0)
print(a/b)

""" 
def foo():
    return 1,2

a = 0
b = 1
a,b+=foo()
print(a,b) """