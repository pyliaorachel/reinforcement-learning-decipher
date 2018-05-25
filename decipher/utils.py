import numpy as np


'''DQN'''

def choose_action_from_dist(actions_value, a_shape):
    subactions_value = split_chunks(actions_value, a_shape)
    return tuple([np.argmax(v) for v in subactions_value])

def choose_action_from_dist_customized(actions_value, a_shape, base, method='one_hot'):
    subactions_value = split_chunks(actions_value, a_shape)
    pure_actions = subactions_value[:-1]
    pred_action = subactions_value[-1]
    if method == 'one_hot':
        return tuple([np.argmax(v) for v in subactions_value])
    elif method == 'ordinal_vec':
        if (pred_action < 0.5).sum() == 0: # If all above threshold, represents the last class
            pred = base - 1
        else:
            pred = np.argmax(pred_action < 0.5)
        return tuple([np.argmax(v) for v in pure_actions] + [pred])
    elif method == 'ordinal_num':
        pred = int(round(pred_action[0] * base))
        return tuple([np.argmax(v) for v in pure_actions] + [pred])

'''List/Array Operations'''

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

def pad_zeros(arr, l):
    return np.array([np.pad(a, ((0, l - len(a)), (0, 0)), 'constant', constant_values=0) for a in arr])

def to_idx_list(l, shape):
    return np.cumsum(np.array([0] + list(shape)[:-1])) + list(l)

'''Symbol Representations & Operations'''

def one_hot(x, size):
    y = np.zeros(size)
    y[int(x)] = 1

    return y

def ordinal_vec(x, size):
    y = np.zeros(size - 1)
    y[range(int(x))] = 1

    return y

def ordinal_num(x, size):
    return np.array([x/size]) 

def one_hot_hstack(x, sizes):
    return np.hstack([one_hot(a, size) for a, size in zip(x, sizes)])

def ordinal_vec_hstack(x, sizes):
    return np.hstack([ordinal_vec(a, size) for a, size in zip(x, sizes)])

def ordinal_num_hstack(x, sizes):
    return np.hstack([ordinal_num(a, size) for a, size in zip(x, sizes)])

def symbol_repr(x, size, method='one_hot'):
    if method == 'one_hot':
        return one_hot(x, size)
    elif method == 'ordinal_vec':
        return ordinal_vec(x, size)
    elif method == 'ordinal_num':
        return ordinal_num(x, size)

def symbol_repr_hstack(x, sizes, method='one_hot'):
    if method == 'one_hot':
        return one_hot_hstack(x, sizes)
    elif method == 'ordinal_vec':
        return ordinal_vec_hstack(x, sizes)
    elif method == 'ordinal_num':
        return ordinal_num_hstack(x, sizes)

def symbol_repr_total_size(sizes, method='one_hot'):
    if method == 'one_hot':
        return sum(sizes) 
    elif method == 'ordinal_vec':
        return sum(sizes) - len(sizes) # For prediction of symbol, ordinal vec is of size = num of classes - 1
    elif method == 'ordinal_num':
        return len(sizes) 
