import torch
import torch.nn.functional as fn
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, embedding_size=10):
        super().__init__()

        self.fc = nn.Linear(input_size, 128)
        self.norm = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.5)
        self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.embed = nn.Embedding(num_embeddings=100, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=64, num_layers=1, batch_first=True)
        self.out = nn.Linear(1024+64, num_classes)

    def forward(self, x, embedded_input):
        x = fn.relu(self.fc(x))
        x = self.norm(x)
        x = self.drop(x)

        x = fn.relu(self.conv(x.unsqueeze(1)))
        x = self.pool(x)
        x = self.flat(x)

        embedded = self.embed(embedded_input)
        lstm, _ = self.lstm(embedded)

        return self.out(torch.cat((x, lstm[:, -1, :]), dim=1))


input = torch.randn(32, 128)
embedding = torch.randint(0, 100, (32, 10))

model = NeuralNetwork(input_size=128, num_classes=10)
outputs = model(input, embedding)
print(outputs.size())
