import matplotlib.pyplot as plt
import torch
import torch.nn.functional as fn


x = torch.linspace(-5, 5, 100)

y_relu = fn.relu(x)
y_sigmoid = fn.sigmoid(x)
y_tanh = fn.tanh(x)
y_leaky_relu = fn.leaky_relu(x)

plt.plot(x, y_relu, label="ReLU")
plt.plot(x, y_sigmoid, label="Sigmoid")
plt.plot(x, y_tanh, label="Tanh")
plt.plot(x, y_leaky_relu, label="LeakyReLU")
plt.legend()
plt.grid()
plt.show()
