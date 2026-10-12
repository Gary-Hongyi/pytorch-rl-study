import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8,2)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x




x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, 3.0],
    [0.0, 1.0, 2.0, 1.0]])

model = MyNet()

target = torch.tensor([[10.0, 20.0],
    [5.0, 8.0],
    [3.0, 6.0]])
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr = 1e-2
)

loss_history = []

for i in range(1000):
    optimizer.zero_grad()
    y = model(x)  
    loss = criterion(y,target)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if(i+1)%50 == 0:
        print(f"Round{i+1},y = {y}, loss = {loss.item()}")
    
print(y)
plt.plot(range(0,1000), loss_history)
plt.xlabel("Training Round")
plt.ylabel("Loss History")
plt.title("Training Curve")
plt.grid()

plt.savefig("mlp_training_multiple_batches1.png",dpi = 150)




