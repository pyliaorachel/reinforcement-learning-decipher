import numpy as np


def space_shape(space):
    if isinstance(space, int):
        return 0
    elif isinstance(space, tuple):
        return [space_shape(s) for s in list(space)]
    else:
        return space.shape

def split_chunks(l, chunk_sizes):
    chunks = []
    i = 0
    for s in chunk_sizes:
        chunks.append(l[i:i+s])
        i += s

    return chunks

def one_hot(x, size):
    y = np.zeros(size)
    y[int(x)] = 1

    return y

def one_hot_hstack(x, sizes):
    return np.hstack([one_hot(a, size) for a, size in zip(x, sizes)])

def pad_zeros(arr, l):
    return np.array([np.pad(a, ((0, l - len(a)), (0, 0)), 'constant', constant_values=0) for a in arr])

def to_idx_list(l, shape):
    return np.cumsum(np.array([0] + list(shape)[:-1])) + list(l)
