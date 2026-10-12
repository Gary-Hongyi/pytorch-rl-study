import torch

w = torch.tensor(5.0, requires_grad=True)

loss = (w - 3) ** 2

loss.backward()

with torch.no_grad():
    w -= 0.1 * w.grad

print(w)

w.grad.zero_()