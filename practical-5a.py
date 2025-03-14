import torch
import torch.nn.functional as fn


y_class_true = torch.tensor([2])
y_class_pred = torch.tensor([[2.0, 1.0, 5.0]])
print(fn.cross_entropy(y_class_pred, y_class_true))

y_log_prob = torch.log_softmax(y_class_pred, dim=1)
print(fn.nll_loss(y_log_prob, y_class_true))

y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.2, 1.9, 3.1])
print(fn.l1_loss(y_pred, y_true))
print(fn.mse_loss(y_pred, y_true))

y_hinge_true = torch.tensor([1, -1, 1])
y_hinge_pred = torch.tensor([0.9, -0.8, 1.2])
print(fn.hinge_embedding_loss(y_hinge_pred, y_hinge_true))
