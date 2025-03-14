import matplotlib.pyplot as plt
import torch


def gradient_descent(f, x, lr=0.1):
    x.grad = None
    y = f(x)
    y.backward()
    with torch.no_grad():
        x -= lr*x.grad

    return x


def f(x):
    return (x-3)**2


EPOCHS = 20

x = torch.linspace(0, 5, 100)
a = torch.tensor(0.0, requires_grad=True)

plt.plot(x, f(x), label="f(x)")
plt.plot(a.detach(), f(a.detach()), "ro", label="Step")

for epoch in range(EPOCHS):
    a = gradient_descent(f, a)
    plt.plot(a.detach(), f(a.detach()), "ro")

plt.xlabel("x")
plt.ylabel("f(x) = (x-3)^2")
plt.legend()
plt.show()
