import tensor
import torch

t_1 = torch.tensor([0, 1], dtype=torch.float, requires_grad=True)
t_2 = torch.tensor([0, 1], dtype=torch.float, requires_grad=True)
t_12 = torch.tensor([0, 0, 0, -1], dtype=torch.float, requires_grad=True)

tn = [t_1, t_2, t_12]
tn_axis = [[1], [2], [1, 2]]

tn = [tensor.get_torch_tensor(tn[t], tn_axis[t]) for t in range(len(tn))]
max_axis = 2

for i in range(1, 3):
    tn, tn_axis = tensor.combine_axis(tn, tn_axis, i)
    tn, tn_axis = tensor.aggregate_axis(tn, tn_axis, i, max_axis)

tn[-1].backward()
print(t_1.grad)
print(t_2.grad)
