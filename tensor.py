import torch


def get_torch_tensor(tensor, axis, max_axis = 1):
    if len(axis) == 0:
        dimension_list = [1 for _ in range(max_axis)]
        return tensor.reshape(tuple(dimension_list))
    dimension_list = []
    for a in range(1, axis[-1] + 1):
        if a in axis:
            dimension_list.append(2)
        else:
            dimension_list.append(1)

    dimension_list.reverse()
    dimension_tuple = tuple(dimension_list)
    return tensor.reshape(dimension_tuple)

def full_contraction_easy(simple_tensors, tn, tn_axis):
    reshaped_tn = [get_torch_tensor(tn[t], tn_axis[t]) for t in range(len(tn))]
    c = combine(reshaped_tn)
    a = aggregate(c)
    a.backward()
    return [simple_tensors[t].grad for t in range(len(simple_tensors))]

def full_contraction_complicated(simple_tensors, tn, tn_axis, max_axis):
    tn = [get_torch_tensor(tn[t], tn_axis[t]) for t in range(len(tn))]
    for i in range(1, max_axis + 1):
        tn, tn_axis = combine_axis(tn, tn_axis, i)
        tn, tn_axis = aggregate_axis(tn, tn_axis, i, max_axis)
    tn[-1].backward()
    return [simple_tensors[t].grad for t in range(len(simple_tensors))]

def combine(tensors):
    t = tensors[0]
    # compute complete combination
    for i in range(1, len(tensors)):
        t = t.add(tensors[i])
    return t

def combine_axis(tn, tn_axis, axis):
    # combine only tensors of a given axis
    new_tensors = []
    new_axis = []
    c_tensors = []
    c_axis = set()
    for i in range(len(tn)):
        if axis in tn_axis[i]:
            c_tensors.append(tn[i])
            c_axis = c_axis.union(tn_axis[i])
        else:
            new_tensors.append(tn[i])
            new_axis.append(tn_axis[i])
    t = c_tensors[0]
    for i in range(1, len(c_tensors)):
        t = t.add(c_tensors[i])
    new_tensors.append(t)
    new_axis.append(list(c_axis))
    return new_tensors, new_axis

def aggregate(tensor):
    l = len(tensor.shape) - 1
    while l > 0:
        tensor = torch.max(tensor, l).values
        l -= 1
    return torch.max(tensor, 0).values

def aggregate_axis(tn, tn_axis, axis, max_axis):
    # aggregate the tensor with axis == axis
    # take last tensor
    t = tn[-1]
    tn.pop(-1)
    tn_axis[-1].remove(axis)

    # aggregate over axis
    t = torch.max(t, max_axis - axis).values
    t = get_torch_tensor(t, tn_axis[-1])

    tn.append(t)
    return tn, tn_axis

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
