import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 1. 定义模型
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# 2. 准备数据

class MyDataset(Dataset):  #和之前最大的不一样是这次是在完整的数据集里面去进行训练
    def __init__(self):
        self.x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, 3.0],
    [0.0, 1.0, 2.0, 1.0]
])
        self.y = torch.tensor([
    [10.0, 20.0],
    [5.0, 8.0],
    [3.0, 6.0]
])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


    


# 3. 创建模型、损失函数、优化器
model = MyNet()
dataSet = MyDataset()

criterion = nn.MSELoss()



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-2
)

loader = DataLoader(
    dataSet,batch_size=2,shuffle=True
)


# 4. 训练
for epoch in range(1000):
    for x_batch,y_batch in loader:

        optimizer.zero_grad()

        prediction = model(x_batch) #每一批拿这一 batch的 x 来进行预测

        loss = criterion(
            prediction,
            y_batch     #预测值和这一批的正确答案来比较算损失
        )

        loss.backward()

        optimizer.step()


        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch+1}, "
                f"Prediction = {prediction}, "
                f"Loss = {loss.item()}"
            )


# 5. 推理阶段
model.eval()

with torch.no_grad():
    result = model(dataSet.x) #最后这步就不取x_batch了，因为是验收看预测，拿整个数据集去验收就行


print("Final prediction:")
print(result)