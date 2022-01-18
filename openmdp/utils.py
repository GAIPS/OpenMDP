import numpy as np

def uniform_distribution(size):
    return np.ones(size) / size

def normalize(array, nan_preventing_eps=1e-8):
    array = array + nan_preventing_eps
    array /= array.sum()
    return array
