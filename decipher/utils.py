"""
Utilities.
"""
import numpy as np


'''RL'''

def space_shape(space):
    """Return the shape of the given space from the environment.
    space: observation or action space from the environment
    """
    if isinstance(space, int):
        return 0
    elif isinstance(space, tuple):
        return [space_shape(s) for s in list(space)]
    else:
        return space.shape

'''DQN'''

def choose_action_from_dist(actions_value, a_shape):
    """Choose an action based on the distribution of action values.
    actions_value: distribution of action values
    a_shape: action shape from the environment
    """
    subactions_value = split_chunks(actions_value, a_shape)
    return tuple([np.argmax(v) for v in subactions_value])

'''List/Array Operations'''

def split_chunks(l, chunk_sizes):
    """Split a list into smaller chunks based on the given list of chunk sizes.
    l: list to split
    chunk_sizes: list of sizes for each chunk
    """
    chunks = []
    i = 0
    for s in chunk_sizes:
        chunks.append(l[i:i+s])
        i += s

    return chunks

def pad_zeros(arr, l):
    """Pad zeros in array elements until the specified length.
    arr: array to pad
    l: total length to fill up to
    """
    return np.array([np.pad(a, ((0, l - len(a)), (0, 0)), 'constant', constant_values=0) for a in arr])

def to_idx_list(l, shape):
    """Turn a list of indices on multiple individual spaces into a list of indices on a single concatenated space.
    l: list of original indices
    shape: shape from the environment
    """
    return np.cumsum(np.array([0] + list(shape)[:-1])) + list(l)

'''Symbol Representations & Operations'''

def one_hot(x, size):
    """One-hot representation.
    x: original value
    size: number of classes
    """
    y = np.zeros(size)
    y[int(x)] = 1

    return y

def ordinal_vec(x, size):
    """Ordinal vector representation.
    x: original value
    size: number of classes
    """
    y = np.zeros(size - 1)
    y[range(int(x))] = 1

    return y

def ordinal_num(x, size):
    """Ordinal number representation.
    x: original value
    size: number of classes
    """
    return np.array([x/size]) 

def one_hot_hstack(x, sizes):
    """Stack a list of values in one-hot representation with their respective number of classes horizontally.
    x: list of original values
    sizes: list of number of classes
    """
    return np.hstack([one_hot(a, size) for a, size in zip(x, sizes)])

def ordinal_vec_hstack(x, sizes):
    """Stack a list of values in ordinal vector representation with their respective number of classes horizontally.
    x: list of original values
    sizes: list of number of classes
    """
    return np.hstack([ordinal_vec(a, size) for a, size in zip(x, sizes)])

def ordinal_num_hstack(x, sizes):
    """Stack a list of values in ordinal number representation with their respective number of classes horizontally.
    x: list of original values
    sizes: list of number of classes
    """
    return np.hstack([ordinal_num(a, size) for a, size in zip(x, sizes)])

def symbol_repr(x, size, method='one_hot'):
    """Return the symbol representation of the given value.
    x: original value
    size: number of classes
    method: symbol representation method
    """
    if method == 'one_hot':
        return one_hot(x, size)
    elif method == 'ordinal_vec':
        return ordinal_vec(x, size)
    elif method == 'ordinal_num':
        return ordinal_num(x, size)

def symbol_repr_hstack(x, sizes, method='one_hot'):
    """Stack a list of values in the specified representation with their respective number of classes horizontally.
    x: list of original values
    sizes: list of number of classes
    method: symbol representation method
    """
    if method == 'one_hot':
        return one_hot_hstack(x, sizes)
    elif method == 'ordinal_vec':
        return ordinal_vec_hstack(x, sizes)
    elif method == 'ordinal_num':
        return ordinal_num_hstack(x, sizes)

def symbol_repr_total_size(sizes, method='one_hot'):
    """Calculate the size of final vector after stacking values of different number of classes in the specified representation.
    sizes: list of number of classes
    method: symbol representation method
    """
    if method == 'one_hot':
        return sum(sizes) 
    elif method == 'ordinal_vec':
        return sum(sizes) - len(sizes) # For prediction of symbol, ordinal vec is of size = num of classes - 1
    elif method == 'ordinal_num':
        return len(sizes) 
