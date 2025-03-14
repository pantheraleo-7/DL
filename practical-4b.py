import torch
import torch.nn.functional as fn
from torch import nn
from torchinfo import summary


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, 32)
        self.lstm = nn.LSTM(32, 64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 128)
        self.norm = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, padding=1, stride=2)
        self.flat = nn.Flatten()
        self.fc2 = nn.Linear(64*8, 10)

    def forward(self, x):
        embedding = self.embed(x)
        lstm, _ = self.lstm(embedding)

        x = lstm[:, -1, :]
        x = fn.relu(self.fc1(x))
        x = self.norm(x)
        x = self.drop(x)

        x = x.unsqueeze(1)
        x = fn.relu(self.conv(x))
        x = self.pool(x)

        x = self.flat(x)
        return self.fc2(x)


VOCAB_SIZE = 100

inputs = torch.randint(0, VOCAB_SIZE, (32, 1024))
model = NeuralNetwork(VOCAB_SIZE)

summary(model, input_data=inputs, col_names=["input_size", "output_size", "num_params"])
