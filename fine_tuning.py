import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models


def evaluate(model, dataloader):
    correct = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, indices = torch.max(outputs, dim=1)
            correct += (indices==labels).sum().item()

    print(f"Accuracy: {correct/len(dataloader.dataset):%}")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

EPOCHS = 1

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters())

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

evaluate(model, testloader)

model.train()
for epoch in range(1, EPOCHS+1):
    total_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS}: Loss = {total_loss/len(trainloader)}")

evaluate(model, testloader)
