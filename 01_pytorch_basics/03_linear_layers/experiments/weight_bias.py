import torch
import torch.nn as nn

layer = nn.Linear(5, 3)

print(layer.weight)
print(layer.weight.shape)

print(layer.bias)
print(layer.bias.shape)

x = torch.tensor([1., 2., 3., 4., 5.])

y = layer(x)

print(y)
print(y.shape)