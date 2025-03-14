import torch


a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(a.T)
print(a+b)
print(a*b)
print(a@b)
print(a.view(1, 4))
print(a.expand(2, -1, -1))
print(a.permute(1, 0))
print(a.narrow(0, 0, 1))
