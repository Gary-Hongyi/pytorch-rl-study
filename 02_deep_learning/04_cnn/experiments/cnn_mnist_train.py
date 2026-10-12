import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


# ---------- 数据 ----------
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ---------- CNN ----------
class CNNNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(
            16 * 7 * 7,
            10
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x


# ---------- 模型 / loss / optimizer ----------
model = CNNNet()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ---------- 训练 ----------
model.train()

for epoch in range(5):

    total_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}, "
        f"Loss = {avg_loss:.4f}"
    )


# ---------- 测试 ----------
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


#可视化来看看到底这个feature map长啥样

# ---------- 可视化第一层 feature maps ----------

# 从测试集中拿一张图片
image, label = test_dataset[0]

# 补上 batch 维
image_input = image.unsqueeze(0)

# 切换到评估模式
model.eval()

with torch.no_grad():

    # 只经过第一层卷积
    feature_maps = model.conv1(image_input)

    # 再经过 ReLU
    feature_maps = model.relu(feature_maps)


print("Feature maps shape:", feature_maps.shape)


# 去掉 batch 维
feature_maps = feature_maps.squeeze(0)


# 创建 2行4列 的画布
fig, axes = plt.subplots(
    2,
    4,
    figsize=(10, 5)
)


# 依次画出8张 feature map
for i, ax in enumerate(axes.flat):

    ax.imshow(
        feature_maps[i],
        cmap="gray"
    )

    ax.set_title(f"Feature {i+1}")

    ax.axis("off")


plt.suptitle(
    f"True label: {label}"
)

plt.tight_layout()

plt.show()





# ---------- 查看第一层卷积核 ----------

# 取出第一层卷积核权重
weights = model.conv1.weight.detach().cpu()

# 看 shape
print("conv1 weight shape:", weights.shape)

# 打印第一个卷积核里的数字
print("Kernel 1 values:")
print(weights[0, 0])


# ---------- 可视化 8 个卷积核 ----------

fig, axes = plt.subplots(2, 4, figsize=(8, 4))

for i, ax in enumerate(axes.flat):

    kernel = weights[i, 0]

    ax.imshow(
        kernel,
        cmap="gray"
    )

    ax.set_title(f"Kernel {i+1}")
    ax.axis("off")


plt.tight_layout()
plt.show()

# ---------- 查看第二层卷积 ----------

print("conv2 weight shape:")
print(model.conv2.weight.shape)


# 取一张测试图片
image, label = test_dataset[0]

# 增加 batch 维
x = image.unsqueeze(0)

model.eval()

with torch.no_grad():

    # 第一层
    x = model.conv1(x)
    x = model.relu(x)
    x = model.pool(x)

    print("After first block:", x.shape)

    # 第二层卷积
    feature_maps2 = model.conv2(x)
    feature_maps2 = model.relu(feature_maps2)

    print("Second feature maps shape:", feature_maps2.shape)


feature_maps2 = feature_maps2.squeeze(0)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for i, ax in enumerate(axes.flat):

    ax.imshow(
        feature_maps2[i],
        cmap="gray"
    )

    ax.set_title(f"Feature {i+1}")
    ax.axis("off")


plt.suptitle(
    f"Conv2 Feature Maps, True label: {label}"
)

plt.tight_layout()
plt.show()