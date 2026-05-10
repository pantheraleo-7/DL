import torch
from torch.utils.data import Dataset, DataLoader


class SquaresDataset(Dataset):
    def __init__(self, start, end):
        self.data = range(start, end)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(x:=self.data[idx]), torch.tensor(x**2)


dataset = SquaresDataset(1, 11)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print("Length of dataset:", len(dataset))
print("Batch Size:", len(dataloader))
