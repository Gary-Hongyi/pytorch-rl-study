import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


# 1. 数据处理
transform = transforms.ToTensor()

#训练集初始化
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


#测试集初始化
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


# 2. 模型

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



model = MNISTNet()


# 3. 损失和优化器

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# 4.训练

for epoch in range(10):

    total_loss = 0


    for images, labels in train_loader:


        optimizer.zero_grad()


        logits = model(images)


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


    print(
    f"Epoch {epoch+1}, "
    f"Loss={total_loss / len(train_loader)}")




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

        total += labels.size(0)

        correct += (prediction == labels).sum().item()


accuracy = correct / total


print(
    f"Test Accuracy: {accuracy*100:.2f}%"
)
# 6. 单张图片预测可视化

model.eval()

image, label = test_dataset[0]


with torch.no_grad():

    # 增加batch维度
    image_input = image.unsqueeze(0)

    # 模型输出
    logits = model(image_input)

    # 转换为概率
    probability = torch.softmax(
        logits,
        dim=1
    )

    # 最大概率对应类别
    prediction = torch.argmax(
        probability,
        dim=1
    )

    # 置信度
    confidence = probability.max()


plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title(
    f"True: {label}, "
    f"Pred: {prediction.item()}, "
    f"Confidence: {confidence.item()*100:.2f}%"
)

plt.show()


#保存模型的参数进行验证
torch.save(
    model.state_dict(),
    "mnist_model.pth"
)

print("Model saved!")





