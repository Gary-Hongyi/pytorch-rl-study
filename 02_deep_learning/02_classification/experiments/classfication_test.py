import torch 
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

x = torch.tensor([
    [1.0, 1.0],
    [1.2, 0.8],
    [4.0, 4.0],
    [4.2, 3.8],
    [8.0, 1.0],
    [8.2, 1.2],
])

y = torch.tensor([0, 0, 1, 1, 2, 2])

model = MyNet()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr = 1e-2
)

model.train()
for epoch in range(200):
    optimizer.zero_grad()

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1}, "
                    f"Prediction = {logits}, "
                    f"Loss = {loss.item()}"
                )


model.eval()
with torch.no_grad():
      logits = model(x)
      prediction = torch.argmax(logits, dim=1)
      accuracy = (prediction == y).float().mean()

print(prediction)
print(accuracy.item())











