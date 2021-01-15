""" This is pyTorch exercises
Exercise 01 : Tensors Explained - Data Structures of Deep Learning

Tensor is a multi dimentional array
"""
"""
Rank : number of dimensions present within the tensor. Suppose we are told to have a rank -2 tensor;
      This means all of following:
      - We have a matrix
      - We have a 2d array
      - We have a 2d Tensor
      
Axes: Number of rank of a Tensor = Number of Axes of the tensor

shape : length of each axes ie number of elements present in each axes of the tensor


"""
# This is my first change in the script

import numpy as np
import torch

# rank 1 tensor

a = [1, 2, 3, 4]
print(a[2])  # rank is 1 because we require only one number of index to extract the element of the tensor a

# rank 2 tensor

aa = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(aa[0][1])  # rank is 2  because we require two numbers of index to extract the element of the tensor a

"""
Axes: Number of rank of a Tensor = Number of Axes of the tensor
"""

# axes example

aa = [[1, 2, 3],   # for this two dimension array first axes = row, second axes = column
      [4, 5, 6],
      [7, 8, 9]]
print(np.sum(aa, axis=0))  # 0 means row
print(np.sum(aa, axis=1))  # 1 means column

t = torch.tensor(aa, dtype = float )


"""
shape : length of each axes ie number of elements present in each axes of the tensor
"""
print(t)
print(t.shape)


# reshape the tensor

tt = torch.tensor([[2, 4, 6],
                   [8, 10, 12],
                   [14, 16, 18]])
print(tt.shape)
print(tt.reshape(1, 9).shape)

# practical exercise

