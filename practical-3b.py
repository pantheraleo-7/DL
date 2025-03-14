import torch


def gradient_descent(f, x, lr=0.1, epochs=50):
    for i in range(1, epochs+1):
        loss = f(x)
        loss.backward()
        with torch.no_grad():
            x -= lr*x.grad
            x.grad.zero_()

        if i%5 == 0:
            print(f"Iteration {i}: x = {x.item()}, Loss = {loss.item()}")

    return x


def f(x):
    return (x - 3) ** 2


a = torch.tensor(0.0, requires_grad=True)

a = gradient_descent(f, a, epochs=10, lr=0.2)
