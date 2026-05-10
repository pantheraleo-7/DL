import sympy as sp
import torch
from sympy.abc import x


def f(x):
    return x**3 + 3*x**2 + 5*x + 7


a = torch.tensor(2.0, requires_grad=True)

df = sp.diff(f(x), x)
grad1 = df.subs(x, a)

f(a).backward()
grad2 = a.grad.item()

print(grad1==grad2)
