import torch

w = torch.tensor(5.0, requires_grad=True)

optimizer = torch.optim.SGD([w], lr=0.1)

loss = (w - 3) ** 2

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(w)