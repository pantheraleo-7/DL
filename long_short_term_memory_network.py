import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(1, 25, num_layers=2, batch_first=True)
        self.fc = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        return self.fc(x)


def train(model, criterion, optimizer, dataloader, epochs=5):
    loss_history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in dataloader:
            optimizer.zero_grad()
            ycap = model(X)
            loss = criterion(ycap, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(total_loss/len(dataloader))

    return loss_history


def generate_sine_data(sequence_length, sample_size=1000):
    p = 2*torch.pi * torch.rand(sample_size).unsqueeze(-1)
    t = torch.linspace(0, (sequence_length-1)*0.1, sequence_length).unsqueeze(0)
    y = torch.sin(t+p).unsqueeze(-1)
    return TensorDataset(y[:, :-1], y[:, -1])


EPOCHS = 5
sequence_lengths = (5, 10, 15)

for seq_len in sequence_lengths:
    dataset = generate_sine_data(seq_len)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    loss_history = train(model, criterion, optimizer, dataloader, EPOCHS)
    plt.plot(range(1, EPOCHS+1), loss_history, label=seq_len)

plt.xticks(range(1, EPOCHS+1))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
