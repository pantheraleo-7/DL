import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*4*4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root="./data", transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

EPOCHS = 5

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model.train()
for epoch in range(1, EPOCHS+1):
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS}: Loss = {total_loss/len(dataloader)}")

filter = model.conv1.weight.data[0]
fig, axes = plt.subplots(1, 3)
for i in range(3):
    axes[i].imshow(filter[i], cmap="gray")
    axes[i].axis("off")
plt.show()

with torch.no_grad():
    feature_maps = model.conv1(dataset[0][0])
fig, axes = plt.subplots(1, 2)
for i in range(2):
    axes[i].imshow(feature_maps[i], cmap="gray")
    axes[i].axis("off")
plt.show()
