import numpy as np


""" a = np.array([[1,2,3],[4,5,6]])
print(a.shape)
b = np.array([0.5,1,2])
print(np.matmul(a,b))
print(np.dot(a,b))
print(a[[1,1],:]) """

a = np.zeros(shape=(5,3))
b = [2,3,3,3,2,2,2]
a[b,:] += 1
print(a)

""" 
def foo():
    return 1,2

a = 0
b = 1
a,b+=foo()
print(a,b) """