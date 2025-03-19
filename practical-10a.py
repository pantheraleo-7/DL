import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self, batchnorm=False, dropout=False, p=0.5):
        super().__init__()

        self.use_dropout = dropout
        self.use_batchnorm = batchnorm

        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        if self.use_dropout:
            self.drop = nn.Dropout(p)
        if self.use_batchnorm:
            self.norm1 = nn.BatchNorm1d(256)
            self.norm2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.norm1(x)
        x = torch.relu(x)
        if self.use_dropout:
            x = self.drop(x)

        x = self.fc2(x)
        if self.use_batchnorm:
            x = self.norm2(x)
        x = torch.relu(x)
        if self.use_dropout:
            x = self.drop(x)

        return self.fc3(x)


def train(model, criterion, optimizer, dataloader, epochs=5, lambda_=0.2, ord=2):
    loss_history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss += lambda_ * sum(p.norm(ord) for p in model.parameters())
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

criterion = nn.CrossEntropyLoss()

model = NeuralNetwork(dropout=True)
optimizer = optim.Adam(model.parameters())
loss_history = train(model, criterion, optimizer, dataloader, EPOCHS)
plt.plot(range(1, EPOCHS+1), loss_history, label="Dropout")

model = NeuralNetwork(batchnorm=True)
optimizer = optim.Adam(model.parameters())
loss_history = train(model, criterion, optimizer, dataloader, EPOCHS)
plt.plot(range(1, EPOCHS+1), loss_history, label="Batch Normalization")

model = NeuralNetwork()
optimizer = optim.Adam(model.parameters())
loss_history = train(model, criterion, optimizer, dataloader, EPOCHS, ord=1)
plt.plot(range(1, EPOCHS+1), loss_history, label="L1 Regularization")

model = NeuralNetwork()
optimizer = optim.Adam(model.parameters())
loss_history = train(model, criterion, optimizer, dataloader, EPOCHS, ord=2)
plt.plot(range(1, EPOCHS+1), loss_history, label="L2 Regularization")

plt.xticks(range(1, EPOCHS+1))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
