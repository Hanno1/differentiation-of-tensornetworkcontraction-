import torch


def get_torch_tensor(tensor, axis):
    dimension_list = []
    for a in range(1, axis[-1] + 1):
        if a in axis:
            dimension_list.append(2)
        else:
            dimension_list.append(1)

    dimension_list.reverse()
    dimension_tuple = tuple(dimension_list)
    return tensor.reshape(dimension_tuple)

def combine(tensors):
    t = tensors[0]
    # compute complete combination
    for i in range(1, len(tensors)):
        t = t.add(tensors[i])
    return t

def aggregate(tensor):
    l = len(tensor.shape) - 1
    while l > 0:
        tensor = torch.max(tensor, l).values
        l -= 1
    return torch.max(tensor, 0).values


if __name__ == "__main__":
    t_1 = torch.tensor([0, 3], dtype=torch.float, requires_grad=True)
    t_2 = torch.tensor([0, 2], dtype=torch.float, requires_grad=True)
    t_12 = torch.tensor([0, 0, 0, -4], dtype=torch.float, requires_grad=True)

    x_1 = get_torch_tensor(t_1, [1])
    x_2 = get_torch_tensor(t_2, [2])
    x_12 = get_torch_tensor(t_12, [1, 2])

    c = combine([x_1, x_2, x_12])
    a = aggregate(c)

    a.backward()

    print(t_1.grad)
    print(t_2.grad)
