"""

exercise on CNN
image = [B, C, H, W]
B : batch size
C : number of color channel, R G B
H : height of the image
W : width of the image

example
img = image[3, 1, 28, 28]
where batch size = 3 batch of three images, each image has single (1) color channel, and height x width is 28x28
color channel = Feature maps
this gives us a rank 4 tensor

"""

import torch
import numpy as np
"""
# the operation  between different data type will cause an error
# example int64 + float32 operations can't be success
# similar way computation  between two devices occur an error 

"""

data = np.array([1, 2, 3])

print(torch.Tensor(data))

print(torch.from_numpy(data))


# Creation options without data

# diagonal element
a = torch.eye(2) # create 2x2 Identity matrix
print(a)

b = torch.zeros(2, 2)  # create 2x2 matrix with all elements are zeros
print(b)

c = torch.ones(2, 2)  # create 2x2 matrix with all elements are ones
print(c)

d = torch.rand(size=(2, 2), dtype=torch.float64)   # create 2x2 matrix with random numbers
print(d)



"""

Flatten, Reshape, and Squeeze 
Flatten: 
Squeeze : tensor is by squeezing and unscrewing them removes all of the axes to have 
          a length of 1 while unsqueezing a tensor adds a dimension with a length of 1
"""

tt = torch.tensor([[1, 1, 1, 1],
                   [2, 2, 2, 2],
                   [3, 3, 3, 3]], dtype=torch.float32)

print(tt.size(), tt.shape)     # both size and shape give us same result.
# The difference is site is a method while shape is a object

print(torch.tensor(tt.shape).prod())  # multiplication of the shapes of the tensor

aa = tt.reshape(2, 2, 3)
print(aa)
print(aa[0][:][:])

# Squeeze

print(tt.reshape(1, 12))
print(tt.reshape(1, 12).shape) # shape = [1, 12]

print(tt.reshape(1, 12).squeeze())
print(tt.reshape(1, 12).squeeze().shape)   # shape = 12 just


# Flatten

def flatten(t):
    t = t.reshape(1, -1)  # -1 ensure unknown number of elements to account for the shape
    t = t.squeeze()

    return t, t.shape


gg = flatten(tt)
print(gg)
print("-------------------------------------------------------------------------------------")
##  --------------------------------------------------------------------------------------------


t1 = torch.tensor([[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]], dtype=torch.float32)

t2 = torch.tensor([[2, 2, 2, 2],
                   [2, 2, 2, 2],
                   [2, 2, 2, 2],
                   [2, 2, 2, 2]], dtype=torch.float32)

t3 = torch.tensor([[3, 3, 3, 3],
                   [3, 3, 3, 3],
                   [3, 3, 3, 3],
                   [3, 3, 3, 3]], dtype=torch.float32)


t = torch.stack((t1, t2, t3))
print(t)
print(t.shape)  # 3x4x4 = 48

t = t.reshape(3, 1, 4, 4)  # 3x1x4x4 = 48
print(t)

# lets flatten this tensor
print(t.reshape(1, -1)[0])
print(t.reshape(-1))  # both gives us the same result
print(t.flatten())   # pytorch default flatten function
print("---------------------------------------------------------")

print(t.flatten(start_dim=1))  # 1 here is the color channel axes
''' this flatten function does flatten on each
image of the batch, however not entire the
images of the batch '''


