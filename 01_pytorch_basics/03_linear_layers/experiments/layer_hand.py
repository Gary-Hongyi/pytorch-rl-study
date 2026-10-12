import torch
import torch.nn as nn

layer = nn.Linear(2, 1)

with torch.no_grad():
    layer.weight.copy_(torch.tensor([[2.0, 3.0]]))
    layer.bias.copy_(torch.tensor([1.0]))

x = torch.tensor([4.0, 5.0])

y = layer(x)

print(y)

target = torch.tensor([30.0])

loss = (y - target) ** 2

print("loss:", loss)

print("before backward:")
print(layer.weight.grad)
print(layer.bias.grad)

loss.backward()

print("after backward:")
print(layer.weight.grad)
print(layer.bias.grad)