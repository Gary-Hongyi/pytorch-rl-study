import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


# 1. 数据处理

transform = transforms.ToTensor()


test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# 2. 定义模型（结构必须和训练时一样）

class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(784,128)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(128,10)


    def forward(self,x):

        x = self.flatten(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x



# 3. 创建空模型

model = MNISTNet()


# 4. 加载训练好的参数

model.load_state_dict(
    torch.load("mnist_model.pth")
)


print("Model loaded successfully!")


# 5. 测试

model.eval()


correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        logits = model(images)

        prediction = torch.argmax(
            logits,
            dim=1
        )


        correct += (
            prediction == labels
        ).sum().item()


        total += labels.size(0)



accuracy = correct / total


print(
    f"Test Accuracy: {accuracy*100:.2f}%"
)