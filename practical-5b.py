import torch
from torch import nn, optim


x = torch.randn(32, 10)
y = torch.randn(32, 1)

optimizers = [
    optim.SGD,
    optim.Adam,
    optim.RMSprop,
    optim.Adagrad
]

for opt in optimizers:
    model = nn.Linear(10, 1)
    criterion = nn.MSELoss()
    optimizer = opt(model.parameters())

    optimizer.zero_grad()
    ycap = model(x)
    loss = criterion(ycap, y)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        ycap = model(x)
        loss = criterion(ycap, y)

    print(f"{optimizer.__class__.__name__} Loss: {loss.item()}")
