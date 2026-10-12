import torch

x=torch.tensor(4.0,requires_grad=True)

y = (x-3)**2

y.backward()

print(x.grad)

y = x ** 2
y.backward()

print(x.grad)

x.grad.zero_()

print(x.grad)      # tensor(0.)