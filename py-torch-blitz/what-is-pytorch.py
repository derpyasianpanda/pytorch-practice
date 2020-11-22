import torch

# Values are whatever was allocated in that memory spot at that time
x_empty = torch.empty(5, 3)
print(x_empty)

x_random = torch.rand(5, 3)
print(x_random)

# dtype just determines the type of the values
x_zeroes = torch.zeros(5, 3, dtype=torch.long)
print(x_zeroes)

# Construction from other data structures
x_construct = torch.tensor([5, 3], dtype=torch.int16)
print(x_construct)

x_changed = x_construct.new_ones(5, 3, dtype=torch.int8)
print(x_changed)

# https://pytorch.org/docs/stable/generated/torch.randn_like.html
# Returns tensor with same size as input and filled w/ random numbers with mean 0
x_changed = torch.randn_like(x_changed, dtype=torch.double)
print(x_changed)

# size() returns a tuple and supports such operations
x_size = x_changed.size()
print(x_size)

# Adding two tensors
x = torch.randint(0, 10, (6, 6), dtype=torch.int8)
print(x)
y = torch.randint(0, 10, (6, 6), dtype=torch.int8)
print(y)
# Two different syntax variations for addition
print(x + y)
print(torch.add(x, y))
# In place addition
x.add_(y) # If a method causes mutation it will be post-fixed with a '_'
print(x)

# You can use NumPy indexing
print(x[:, 1])

# Reshape with torch.view
x_reshaped = x.reshape(36)
x_auto_reshaped = x.reshape(-1, 12) # -1 indicated PyTorch must infer size
print(x.size(), x_reshaped.size(), x_auto_reshaped.size())

# Retrieve value from 1x1 tensors
x_single = torch.randn(1) # Post fix n usually means normal distribution rather than uniform
print(x_single, x_single.item())

# When using alongside NumPy, memory is often shared
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
c = b.numpy()
print(a, b ,c)
np.add(a, 1, out=a)
print(a, b, c)

# We will use torch.device to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # .to can also change dtype together!