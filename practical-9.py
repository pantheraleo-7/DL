import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, 128)
        self.tf = nn.Transformer(128, nhead=16, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, src, tgt):
        x = self.tf(self.embed(src), self.embed(tgt))
        return self.fc(x)

    def greedy_decode(self, src, max_len=5):
        self.eval()
        tgt = torch.zeros((src.size(0), 1), dtype=torch.long)

        for _ in range(max_len-1):
            outputs = self(src, tgt)
            next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat((tgt, next_token), dim=1)

        return tgt

    def beam_search_decode(self, src, beam_width=3, max_len=5):
        self.eval()
        tgts_and_scores = [(torch.zeros((src.size(0), 1), dtype=torch.long), 0)]

        for _ in range(max_len-1):
            candidates = []
            for tgt, score in tgts_and_scores:
                outputs = self(src, tgt)
                probs = torch.log_softmax(outputs[:, -1, :], dim=-1)
                topk = torch.topk(probs, beam_width, dim=-1)

                for i in range(beam_width):
                    new_tgt = torch.cat((tgt, topk.indices[:, i].unsqueeze(1)), dim=1)
                    new_score = score+topk.values[:, i].item()
                    candidates.append((new_tgt, new_score))

            tgts_and_scores = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        return tgts_and_scores[0][0]


def generate_seq2seq_data(vocab_size, seq_length=5, num_samples=1000):
    data = torch.randint(1, vocab_size, (num_samples, seq_length))
    target = data.flip(dims=[1])
    return TensorDataset(data, target)


EPOCHS = 5
VOCAB_SIZE = 20

dataset = generate_seq2seq_data(VOCAB_SIZE)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Transformer(VOCAB_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model.train()
for epoch in range(1, EPOCHS+1):
    total_loss = 0
    for src, tgt in dataloader:
        optimizer.zero_grad()
        outputs = model(src, tgt[:, :-1])
        loss = criterion(outputs.flatten(end_dim=-2), tgt[:, 1:].flatten())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS}: Loss = {total_loss/len(dataloader)}")

sample_input = torch.randint(1, VOCAB_SIZE, (1, 5))
greedy_output = model.greedy_decode(sample_input)
beam_output = model.beam_search_decode(sample_input)

print("Input Sequence:", sample_input)
print("Greedy Decoded Output:", greedy_output)
print("Beam Search Decoded Output:", beam_output)
