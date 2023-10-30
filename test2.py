import torch


t = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.float, requires_grad=True)
print(torch.max(t, 0).values)
print(torch.max(t, 1).values)
print(torch.max(t, 2).values)
