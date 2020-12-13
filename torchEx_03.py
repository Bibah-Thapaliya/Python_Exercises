"""
Broadcasting and Element-wise operation

An element wise operation is an operation between two tensors that operates on corresponding elements within the
the respective tensor.


"""

import torch
import numpy as np

t1 = torch.tensor([[1, 2],
                   [3, 4]], dtype=torch.float32)

t2 = torch.tensor([[9, 8],
                   [7, 6]], dtype=torch.float32)

# same shapes requires in order to perform the element wise operation

# addition

print(t1 + t2)
print(t1 - 2)  # scalar value tensor  is being broadcasted as to the shape of tensor t1, and
# then the element wise operation is carried out

print(t2 * 2)
print(t2 / 2)

# broadcast_to() numpy function

a = np.broadcast_to(2, t1.shape)
print(a)

b = t1 + torch.tensor(np.broadcast_to(2, t1.shape), dtype=torch.float32)  # so scalar value 2 is broadcasted
# to just like rank 2 tensor t1
print(b)
print("--------------------------------------------------------------------------------------")


t3 = torch.tensor([[1, 2],
                   [3, 4]], dtype=torch.float32)

t4 = torch.tensor([3, 4], dtype=torch.float32)

# t1 +t2 ???
# even though the shapes of t3 and t4 are not same
# using numpy broadcast function we can perform
# the element wise operation

t4 = np.broadcast_to(t4.numpy(), t3.shape)
print(t3 + t4)
print("--------------------------------------------------------------------------------------")

"""

Tensor reduction operation:
A reduction operation on a tensor is an operation that reduces the number of 
elements contained within the tensor


"""

t = torch.tensor([
    [0, 1, 0],
    [2, 0, 2],
    [0, 3, 0]
], dtype=torch.float32)

print(t.sum())
print(t.numel())  # checking number of element
# product
print(t.prod())
# mean
print(t.mean())
# standard deviation
print(t.std())

a = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]], dtype=torch.float32)

print(a.sum(dim=0))  # sum in row wise
print(a.sum(dim=1))  # sum in column wise
print(a[1].sum())   # second row's elements addition
print("-----------------------------------------------------------------")

"""
ArgMax tensor reduction operation
It tells us which argument when supplied to a function as input results
in the functions max output value.

So Argmax tells us the index location of the maximum value inside a tensor.
"""

d = torch.tensor([
    [0, 1, 0],
    [2, 0, 2],
    [0, 3, 0]], dtype=torch.float32)
print(d.max())  # this confirms which is the maximum value
print(d.flatten())
print(d.argmax())  # this confirms the location of maximum value
# location of maximum value is as per the flatten operation

print(d.max(dim=0))  # search for maximum value along rows, this also returns the index of the maximum value
print(d.max(dim=1))  # search for maximum value along column

print("++++++++++++++++++++++++++++++++++++++++++")

print(d.mean())
print(d.mean(dim=0).tolist())  # tolist() function helps us to get the items of the tensor
print(d.mean(dim=0).numpy())  # converting the tensor into the numpy array
