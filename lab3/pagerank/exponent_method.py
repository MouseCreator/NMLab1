import numpy as np


def exp_method(matrix, vector, coord, eps):
    if coord > len(vector):
        raise ValueError("M coordinate must fit in vector!")
    if np.abs( vector[coord] ) < eps:
        raise ValueError("The leading coordinate is zero")
    x_prev = vector.copy()
    while True:
        x_curr = np.dot(matrix, x_prev)
        max_eigen_value = x_curr[coord] / x_prev[coord]
        if np.linalg.norm(x_curr - x_prev) < eps:
            return max_eigen_value
        x_prev = x_curr.copy()


def find_max_eigen_value(matrix, eps):
    dim = matrix.shape[0]
    vector = np.ones(dim)
    coord = 0
    return exp_method(matrix, vector, coord, eps)