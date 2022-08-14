import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

## Daniel's Notebook
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/00_pytorch_fundamentals.ipynb

# ------------------------------------------------------------------------------
## Introduction to Tensors
# ------------------------------------------------------------------------------
### Creating tensors
# PyTorch tensors are created using torch.Tensor()
# https://pytorch.org/docs/stable/tensors.html

## scalar
scalar = torch.tensor(7)
scalar

# number of dimensions
scalar.ndim

# get python data
scalar.item()

## vector
vector = torch.tensor([7, 7])
vector

vector.ndim

vector.shape

## matrix
MATRIX = torch.tensor([[7, 8], [9, 10]])
MATRIX

MATRIX.ndim

MATRIX.shape

MATRIX = torch.tensor([[7, 8, 9], [9, 10, 11]])
MATRIX.shape

## TENSOR
TENSOR = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
TENSOR.ndim
TENSOR.shape
TENSOR[0][0]
TENSOR[0][1]
TENSOR[0][2]
TENSOR[0][2][0]

# scalars and vectors usually use lowercase variables
# matricies and tensors usually use uppercase variables

### Random tensors

# Create a random tensor
# https://pytorch.org/docs/stable/generated/torch.rand.html

random_tensor = torch.rand(3, 4)
random_tensor
random_tensor.ndim
random_tensor.shape

# Create a random tensor with similar shape to an image
# height width colors
random_image_tensor = torch.rand(size=(224, 224, 3)) 
random_image_tensor

### Zeros and ones
zeros = torch.zeros(size=(3, 4))

zeros * random_tensor

ones = torch.ones(size=(3, 4))
ones


### Creating a range of tensors and tensors-like
# https://pytorch.org/docs/stable/generated/torch.arange.html

# Use torch.arange()
torch.arange(0, 10)

torch.arange(start=0, end=10, step=2)

# Create tensors like
like_ones = torch.zeros_like(ones)
like_ones
like_ones.shape

# https://youtu.be/Z_ikDlimN6A?t=6822

# ------------------------------------------------------------------------------
### Tensor datatypes
# ------------------------------------------------------------------------------
# Datatypes is one of the 3 big errors you'll run into with PyTorch & deep learning
# 1. Tensors not right datatype
# 2. Tensors not right shape
# 3. Tensors not on right device

# https://pytorch.org/docs/stable/tensors.html
# default is float32

float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None)
float_32_tensor

float_32_tensor = torch.tensor(
    [3.0, 6.0, 9.0],
    dtype=None,
    device=None, # "cpu", "cuda"
    requires_grad=False) # do you want gradient tracking

float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor

int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
int_32_tensor

# Daniel doesn't know the rules for multipling different datatypes nor do I
# this is fine 
float_16_tensor * int_32_tensor

## Getting info from tensors (attributes)

some_tensor = torch.rand(3, 4)
some_tensor
some_tensor.dtype
some_tensor.shape
some_tensor.size()
some_tensor.device

# ------------------------------------------------------------------------------
### Tensor operations
# ------------------------------------------------------------------------------
# https://youtu.be/Z_ikDlimN6A?t=8077
# - Addition
# - Subtraction
# - Division
# - Element-wise multiplication
# - Matrix multiplication

tensor = torch.tensor([1, 2, 3])
tensor * 10
tensor / 10
tensor + 10
tensor - 10

# PyTorch built-in functions
torch.mul(tensor, 10)
torch.div(tensor, 10)
torch.add(tensor, 10)
torch.sub(tensor, 10)

# ------------------------------------------------------------------------------
## Matrix multiplication (dot product)
# ------------------------------------------------------------------------------
# http://www.mathsisfun.com/algebra/matrix-multiplying.html
# http://matrixmultiplication.xyz

print(tensor, "*", tensor)
print(f"Equals: {tensor * tensor}")

# 1*1 + 2*2 + 3*3
torch.matmul(tensor, tensor)
tensor @ tensor

value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]

## One of the most common erros in deep learning is a shape error
### Matrix multiplication rules
# 1. Inner dimensions must match
# shapes
# (3, 2) @ (3, 2) error
# (2, 3) @ (3, 2) yes
# (3, 2) @ (2, 3) yes
# 2. The resulting matrix has the shap of the outer dimensions
# (2, 3) @ (3, 2) -> (2, 2)

shapely = torch.matmul(torch.rand(2, 3), torch.rand(3, 2))
shapely
shapely.shape

shapely = torch.matmul(torch.rand(3, 2), torch.rand(2, 3))
shapely
shapely.shape

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])

# Transpose (vertical to horizontal)
tensor_A
tensor_A.T

torch.matmul(tensor_A, tensor_B.T)
torch.matmul(tensor_A.T, tensor_B)

# ------------------------------------------------------------------------------
## Aggregation
# ------------------------------------------------------------------------------

tensor_A = torch.tensor(tensor_A, dtype=torch.float32)
# or
tensor_A = tensor_A.type(torch.float32)

tensor_A.max()
tensor_A.min()
tensor_A.mean()
tensor_A.sum()

torch.max(tensor_A)
torch.min(tensor_A)
torch.mean(tensor_A)
torch.sum(tensor_A)

## Argmax (positional max)

tensor_A.argmin()
tensor_A.argmax()

torch.argmin(tensor_A)
torch.argmax(tensor_A)

# ------------------------------------------------------------------------------
## Reshaping, stacking, squeezing and unsqueezing
# ------------------------------------------------------------------------------
# https://youtu.be/Z_ikDlimN6A?t=10766
# - Reshape - reshape an input tensor
# - View - Return a view of an input tensor of certain shape but shares the same
# memory of the original tensor
# - Stack - concatenate a sequence of tensors along a new dimension
# hstack, vstack, stack
# - Squeeze - remove all '1' dimensions from a tensor
# - Unsqueeze - and a '1' dimension to a target tensor
# - Permute - return a view of the input with dimensions permuted (swapped) in
# a certain way

# Reshape - add an extra dimension
x = torch.arange(1., 10.)
x
x.shape

x_reshaped = x.reshape(1, 9)
x_reshaped

x_reshaped = x.reshape(9, 1)
x_reshaped

x_reshaped = x.reshape(3, 3)
x_reshaped


x = torch.arange(1., 13.)
x_reshaped = x.reshape(3, 4)
x_reshaped

# Change the view
x = torch.arange(1., 10.)
z = x.view(1, 9)
z

## observe the shared memory
x
z[:, 0] = 5
z, x

# Stack tensors
rando = torch.rand(9)

x_stacked = torch.stack([x, x, x, rando])
x_stacked

x_stacked = torch.stack([x, x, x, rando], dim=1)
x_stacked

x_stacked = torch.stack([x, x, x, rando], dim=-1)
x_stacked

x_stacked = torch.stack([x, x, x, rando], dim=-2)
x_stacked

x_stacked = torch.hstack([x, x, x, rando])
x_stacked

x_stacked = torch.vstack([x, x, x, rando])
x_stacked

# ??? vstack hstack

# Squeeze - remove the outer dimension

x = torch.arange(1., 10.)
x_reshaped = x.reshape(1, 9)
x_reshaped
torch.squeeze(x_reshaped)

# Unsqueeze - add a dimension at dim

x = torch.arange(1., 10.)
x_reshaped = x.reshape(3, 3)
print(x_reshaped.numpy())
print()
print(torch.unsqueeze(x_reshaped, dim=0).numpy())
print()
print(torch.unsqueeze(x_reshaped, dim=1).numpy())
print()
print(torch.unsqueeze(x_reshaped, dim=2).numpy())
print()

# Permute - a view on the memory

x = torch.rand(size=(224, 448, 3))
x.shape
x_permuted = torch.permute(x, dims=(2, 0, 1))
x_permuted.shape

## notice the shared memory

x[0, 0, 0]
x_permuted[0, 0, 0]
x[0, 0, 0] = 123456
x_permuted[0, 0, 0]

# Indexing with PyTorch

x = torch.arange(1, 10).reshape(1, 3, 3)
x
x[0]
x[0, 0]
x[0, 1, 2]

x[0, :, 2]
x[:, :, 2]

x[0, 0:2, 1]

# ------------------------------------------------------------------------------
# PyTorch and Numpy
# ------------------------------------------------------------------------------

## default numpy dtype is np.float64
## default torch dtype is torch.float32
## from_numpy will take the dtype of the numpy array
n = np.arange(1., 8.)
x = torch.from_numpy(n)
x
## changing the dtype
x = torch.from_numpy(n).type(torch.float32)
x
# or
n = np.arange(1., 8., dtype=np.float32)
x = torch.from_numpy(n)
x

n = np.arange(1., 8.)
x = torch.from_numpy(n)
n2 = x.numpy()
n2.dtype
x = torch.from_numpy(n).type(torch.float32)
n3 = x.numpy()
n3.dtype

# https://youtu.be/Z_ikDlimN6A?t=13305

# ------------------------------------------------------------------------------
# Reproducability (take out the random)
# ------------------------------------------------------------------------------
# https://pytorch.org/docs/stable/notes/randomness.html

## seed the random number generator for all device (CPU, and CUDA)
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random_a = torch.rand(3, 4)
print(random_a)
random_b = torch.rand(3, 4)
print(random_b)

torch.manual_seed(RANDOM_SEED)
random_1 = torch.rand(3, 4)
print(random_1)
print(random_a == random_1)
random_2 = torch.rand(3, 4)
print(random_2)
print(random_a == random_1)

## for custom operators, you might need to set python seed as well
import random
random.seed(RANDOM_SEED)

## if you depend on NumPy, seed the global Numpy RNG
import numpy as np
np.random.seed(RANDOM_SEED)

## Running on GPU
# Use Google Colab
# Use your own GPU, read https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/
# use cloud computing - GCP, AWS, Azure

torch.cuda.is_available()

## Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)

# count number of devices
torch.cuda.device_count()

# create a tensor (default on the CPU)
# https://youtu.be/Z_ikDlimN6A?t=14773
tensor = torch.tensor([1, 2, 3], device="cpu")

# move tensor to gpu (if available)
tensor_on_gpu = tensor.to(device="cuda")

tensor_on_gpu = tensor.to(device)
tensor_on_gpu.device

## tensor must be on cpu to transform to numpy

## Practice exercises
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/00_pytorch_fundamentals_exercises.ipynb





