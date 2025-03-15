import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train(model, criterion, optimizer, dataloader, epochs=5):
    loss_history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(total_loss/len(dataloader))

    return loss_history


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="./data", transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

EPOCHS = 5
optimizers = [
    optim.SGD,
    optim.Adam,
    optim.RMSprop
]

for opt in optimizers:
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = opt(model.parameters())

    loss_history = train(model, criterion, optimizer, dataloader, EPOCHS)
    plt.plot(range(1, EPOCHS+1), loss_history, label=optimizer.__class__.__name__)

plt.xticks(range(1, EPOCHS+1))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
