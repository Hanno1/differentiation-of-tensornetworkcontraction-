import torch
import time
import random
import itertools as it


def get_torch_tensor(tensor, axis, max_axis = 1):
    axis = sorted(axis)
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

def full_contraction_easy(simple_tensors, tn, tn_axis, t=False):
    reshaped_tn = [get_torch_tensor(tn[t], tn_axis[t]) for t in range(len(tn))]
    c = combine(reshaped_tn)
    a = aggregate(c)
    if t:
        start = time.time()
        a.backward()
        end = time.time()
        return [simple_tensors[t].grad for t in range(len(simple_tensors))], end - start
    a.backward()
    return [simple_tensors[t].grad for t in range(len(simple_tensors))]

def full_contraction_complicated(simple_tensors, tn, tn_axis, max_axis, t=False):
    tn = [get_torch_tensor(tn[t], tn_axis[t]) for t in range(len(tn))]
    for i in range(1, max_axis + 1):
        tn, tn_axis = combine_axis(tn, tn_axis, i)
        tn, tn_axis = aggregate_axis(tn, tn_axis, i, max_axis)
    if t:
        start = time.time()
        tn[-1].backward()
        end = time.time()
        return [simple_tensors[t].grad for t in range(len(simple_tensors))], end - start
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
    # get max axis in tensor
    a = len(t.shape)
    # t = torch.max(t, max_axis - axis).values
    t = torch.max(t, a - axis).values
    t = get_torch_tensor(t, tn_axis[-1])

    tn.append(t)
    return tn, tn_axis

def create_full_tensornetwork(max_axis, max_range=None, lower=-10, upper=10):
    # create big tensornetwork
    axis_list = [str(i + 1) for i in range(max_axis)]

    if max_range is None:
        max_range = max_axis

    # create 1 dimensional tensors
    tn = []
    tn_axis = []
    simple_tensors = []
    for i in range(max_axis):
        t = torch.tensor([0, random.randint(lower, upper)], dtype=torch.float32, requires_grad=True)
        tn.append(t)
        simple_tensors.append(t)
        tn_axis.append([i + 1])

    # create 2 dimensional tensors
    for i in range(2, max_range + 1):
        comb = list(it.combinations(axis_list, i))
        for p in comb:
            l = [0 for _ in range(2**i)]
            l[-1] = random.randint(lower, upper)
            t = torch.tensor(l, dtype=torch.float32, requires_grad=True)
            tn.append(t)

            axis = [int(i) for i in list(p)]
            tn_axis.append(axis)

    return simple_tensors, tn, tn_axis

def create_sparse_tensornetwork(max_axis, lower=-10, upper=10):
    # create 1 dimensional tensors
    tn = []
    tn_axis = []
    simple_tensors = []
    for i in range(max_axis):
        t = torch.tensor([0, random.randint(lower, upper)], dtype=torch.float32, requires_grad=True)
        tn.append(t)
        simple_tensors.append(t)
        tn_axis.append([i + 1])

    # create 2 dimenstional tensors -> axis that will be connected: 1,2 . 2,3 . 3,4 ...
    for axis in range(1, max_axis):
        l = [0 for _ in range(2**2)]
        l[-1] = random.randint(lower, upper)
        t = torch.tensor(l, dtype=torch.float32, requires_grad=True)
        tn.append(t)
        tn_axis.append([axis, axis + 1])
    return simple_tensors, tn, tn_axis
    

def compare_algorithms(n, max_axis, create_tn):
    t_1 = 0
    d_1 = 0
    t_2 = 0
    d_2 = 0
    for i in range(n):
        print("Starting Iteration ", i)
        simple_tensors, tn, tn_axis = create_tn(max_axis)

        start = time.time()
        _, deriv_1 = full_contraction_easy(simple_tensors, tn, tn_axis, t=True)
        end = time.time()
        d_1 += deriv_1
        t_1 += end - start

        start = time.time()
        _, deriv_2 = full_contraction_complicated(simple_tensors, tn, tn_axis, max_axis, t=True)
        end = time.time()
        d_2 += deriv_2
        t_2 += end - start

    t_1 /= n    
    t_2 /= n
    d_1 /= n
    d_2 /= n

    return t_1, d_1, t_2, d_2

if __name__ == "__main__":
    compare_algorithms(1, 8)
