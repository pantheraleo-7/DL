import torch


x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 3*x**2 + 5*x + 7

y.backward()

print(x.grad)
