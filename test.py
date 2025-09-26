import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义预处理
transform = transforms.Compose([
    transforms.ToTensor(),                # 转为张量，像素缩放到 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化，常用均值和标准差
])

# 加载训练集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# 加载测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

# 使用 DataLoader 批量加载
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1000, shuffle=False)

# 查看一个 batch
images, labels = next(iter(train_loader))
print(images.shape)  # torch.Size([64, 1, 28, 28])
print(labels.shape)  # torch.Size([64])

# 取一个 batch
images, labels = next(iter(train_loader))

# 反归一化函数
def denormalize(img):
    return img * 0.3081 + 0.1307

# 显示一个批次（64 张图像）
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    img = denormalize(images[i]).squeeze().numpy()
    ax.imshow(img, cmap="gray")
    ax.set_title(str(labels[i].item()), fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()
