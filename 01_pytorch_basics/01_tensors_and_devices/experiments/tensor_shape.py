import torch

x =torch.arange(12)

print(x)
print(x.shape)

x = x.reshape(3,-1)
print(x)
x = x.reshape(-1,6)
print(x)

a = torch.tensor([1,2,3])
b=a.unsqueeze(0)
print(b)
print(b.shape)
c=a.unsqueeze(1)
print(c)
print(c.shape)

state=torch.tensor([0.2,0.5,0.8,1.0])
state=state.unsqueeze(1)
print(state)
print(state.shape)