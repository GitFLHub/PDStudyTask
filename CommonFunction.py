import numpy as np


def normalization(array):
    return (array - array.mean(axis=0)) / (np.max(array, axis=0) - np.min(array, axis=0)) / np.std(array, axis=0)


def multiType(array):
    type = np.unique(array)
    m = array.shape[0]
    res = np.zeros((m, len(type)))
    dict = {}
    for i in np.arange(m):
        res[i, np.where(type == array[i])] = 1
    for j in np.arange(len(type)):
        dict[str(j)] = type[j]
    return [res, dict]
