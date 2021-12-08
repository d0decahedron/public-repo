import torch

#########################################################
# YOUTUBE TUTORIAL TENSORS
########################################################


device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32, device=device, requires_grad=True)

#print(my_tensor)
#print(my_tensor.dtype)
#print(my_tensor.device)
#print(my_tensor.shape)


# Other initialization methods

x = torch.empty(size=(3, 3))
x = torch.zeros((3,3))
x = torch.rand((3,3))  # random initialization between 0 and 1
x = torch.ones((3,3))
x = torch.eye(5,5)   # identity matrix
#print(x)

x = torch.arange(start=0, end=5, step=1)
#print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
#print(x)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1) # make values normal distributed
x = torch.empty(size=(1,5)).uniform_(0,1)
x = torch.diag(torch.ones(3))

# How to initialize and convert to different types

tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long()) # int64
print(tensor.half())
print(tensor.float()) # float32
print(tensor.double())

# array to tensor
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()


# Tensor Math
x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

z1 = torch.empty(3)
torch.add(x, y, out= z1)
print(z1)
z2 = torch.add(x, y)
z = x + y

z = x - y
print(z)

# division
z = torch.true_divide(x, y)
print(z)
u = torch.true_divide(x, 2)
print(u)

t = torch.zeros(3)
t.add_(x) #equal to t += x
print(t)

z = x.pow(2)
print(z)
z = x ** 2 # the same

# Comparisons
z = x > 0
print(z)
z = x < 0
print(z)

# Matrix multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1, x2)
# equivalent
x3 = x1.mm(x2)


# matrix exponentiation
matrix_exp = torch.rand((5,5))
print(matrix_exp.matrix_power(3))

# elementwise multiplication
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # (batch, n, p)
print(out_bmm.shape)

# broadcasting
x1 = torch.rand((5,5))
print(x1)
x2 = torch.rand((1,5))
print(x2)
z = x1 - x2     # copies the 1x5 tensor and substracts it from each row of the matrix
print(z)

z = x1 ** x2 #

sum_x = torch.sum(x, dim=0)
print(sum_x)

values, indices = torch.max(x, dim=0)
print(values, indices)
values, indices = torch.min(x, dim=0)
print(values, indices)
abs_x = torch.abs(x)

z = torch.argmax(x, dim=0) # similar to above but only returns the index and not the value

mean_x = torch.mean(x.float(), dim=0)
print(mean_x)

z = torch.eq(x,y) # which elements are equal
sorted_y, indices = torch.sort(y, dim=0, descending=False)
print(y)
print(sorted_y)

z = torch.clamp(x, min=0, max=10) # sends values <0 to 0 and values >10 to 10
print(z)

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
print(x)
z = torch.any(x)
print(z)
z = torch.all(x)
print(z)

# Indexing
batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0]) # same as x[0,:]
print(x[0].shape)

print(x[:, 0]) # first column of all rows
print(x[:,0].shape)

print(x[2, 0:10])  #0:10 -> [0,1,2,...,9], so third row and all ten columnes

x[0,0] = 100

# fancy indexing
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x = torch.rand((3,5))
print(x)
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows, cols])

# advanced indexing
x = torch.arange(10)
print(x[(x < 2 ) | (x > 8)])
print(x[(x < 2 ) & (x > 8)])
print(x[x.remainder(2) == 0])

# useful operations
print(x)
print(torch.where(x > 5, x, x*2))
print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension()) # if 5x5x5 then ndimension is 3
print(x.numel()) # count number of elements

# Reshaping
x = torch.arange(9)
x_3x3 = x.view(3,3) # needs to be contiguous
print(x)
print(x_3x3)
print(x_3x3.shape)

x_3x3 = x.reshape(3,3) # can always work but can be less performant
print(x_3x3.shape)

y = x_3x3.t()
print(y)
#print(y.view(9)) # does not work
print(y.contiguous().view(9))
#print(y.reshape(9)) does also work


x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim=0).shape)
print(torch.cat((x1,x2), dim=1).shape)

z = x1.view(-1)
print(z.shape)

batch = 64
x  = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

# switch the last two dimensions but keep batch as first
z = x.permute(0, 2, 1)
print(z.shape)

x = torch.arange(10) # [10]
print(x.unsqueeze(0))

print(x.unsqueeze(1))

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10
print(x.shape)

z = x.squeeze(1)
print(z.shape)