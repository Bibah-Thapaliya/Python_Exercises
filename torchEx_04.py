"""
Matrix multiplication operation
"""

import torch
import numpy as np

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# addition and subtraction
z1 = torch.empty(3)
print(z1)
torch.add(x, y, out=z1)

# Or
z2 = torch.add(x, y)
print(z2)

# or
print(x + y)
print("-----------------------")


# Division

z = torch.true_divide(x, y)  # element wise division if x, y are of same shapes
print(z)

# Exponentiation

z = x.pow(2) # element wise power operation
print(z)
print(z**2)  # same operation without calling the pow() function


# simple comparison element wise

z = x > 0
print(z)

# matrix multiplication
x1 = torch.rand(size=(2, 5), dtype=torch.float32)
x2 = torch.rand(size=(5, 3), dtype=torch.float32)
x3 = torch.mm(x1, x2)  # output shape 2x3
print(x3)

# matrix exponentation (not element wise)
m = torch.rand(size=(5, 5), dtype=torch.float32)
print(m)
print(m.matrix_power(3))

# dot product
# dot product is first element wise multiplication and the sum
z = torch.dot(x, y)
print(z)

# batch matrix multiplication
batch = 32  # batch site
n = 10  # number of rows
m = 20  # number of columns of tensor1, and number of rows of tensor2
p = 30  # number of columns of tensor2
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
#  print(tensor1)
#  print(len(tensor1))

out_bmm = torch.bmm(tensor1, tensor2) # size of the output matrix is (batch, n, p)
# print(out_bmm)

# Example of Broadcasting
# Broadcasting happens in element wise
x1 = torch.rand(size=(5, 5), dtype=torch.float32)
x2 = torch.rand(size=(1, 5), dtype=torch.float32)

z = x1 - x2
print(z)

