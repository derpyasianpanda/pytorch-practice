import torch

# Autograd is a useful package that allows for automatic operations to be run
# on your network
# Autograd is basically an engine for computing Vector Jacobian product

# Create a tensor that tracks computation
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = 3 * y * y
print(z, z.mean())

# You can change if you want to track computation
a = torch.rand(2, 2)
a = a ** 2 + 1
print(a, a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a ** 3).sum()
print(b, b.grad_fn)

# Backpropagation and Gradients
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = 3 * y * y
out = z.mean()
out.backward()
# Essentially d(out)/dx
print(x.grad)

# Since y is not a scalar, we cannot compute full Jacobian directly. But we can get the product
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# You can also prevent gradient (for testing or etc)
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

# You can also detach to get a copy of the tensor that doesn't have gradient
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())