
import torch

a = torch.ones(3)
a = torch.diag(a)
b = a*a

print(b)