import torch

x = torch.randn(3,3)

print(x)
print(x.shape)
print(x.dtype)

a=torch.tensor([1,2,3])
b=torch.tensor([4,5,6])

print(a+b)
print(a*b)
print(a.device)
a=a.to("cuda")
print(a.device)

