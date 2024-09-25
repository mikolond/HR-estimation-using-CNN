import torch
import numpy as np

a = np.array([[1,2],[2,3]])
x = np.array([3,4]).T
a = torch.Tensor(a)
x = torch.Tensor(x)

y = torch.matmul(a,x)

print("y:",y)