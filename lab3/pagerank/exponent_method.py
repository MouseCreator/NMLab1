import numpy as np

def exp_method(matrix, vector, coord, eps):
    if coord > len(vector):
        raise ValueError("M coordinate must fit in vector!")
    if np.abs(vector[coord]) < eps:
        raise ValueError("The leading coordinate is zero")
    x_prev = vector
    max_eigen_value = 0
    while True:
        x_curr = np.dot(matrix, x_prev)
        lambda_prev = max_eigen_value
        max_eigen_value = x_curr[coord] / x_prev[coord]
        diff = np.abs(max_eigen_value - lambda_prev)
        if diff < eps:
            return max_eigen_value
        x_prev = x_curr.copy()


def exp_method_vector(matrix, vector, eps):
    if np.all(vector == 0):
        raise ValueError("Vector cannot be zero vector")
    while True:
        b_k1 = np.dot(matrix, vector)
        b_k1_norm = np.linalg.norm(b_k1)
        if (np.linalg.norm(b_k1 / b_k1_norm - vector)) < eps:
            return b_k1
        vector = b_k1 / b_k1_norm


def exp_scalar_method(matrix, vector, eps):
    if np.all(vector == 0):
        raise ValueError("Vector cannot be zero vector")
    x_prev = vector.copy()
    max_eigen_value = 0
    while True:
        e_temp = x_prev / np.linalg.norm(x_prev)
        x_curr = np.dot(matrix, e_temp)
        lambda_prev = max_eigen_value
        max_eigen_value = np.inner(x_curr, e_temp)
        if np.abs(max_eigen_value - lambda_prev) < eps:
            return max_eigen_value
        x_prev = x_curr.copy()


def find_max_eigen_value(matrix, eps):
    dim = matrix.shape[0]
    vector = np.ones(dim)
    coord = 0
    return exp_method(matrix, vector, coord, eps)


def find_eigen_vector(matrix, eps):
    dim = matrix.shape[0]
    vector = np.ones(dim)
    return exp_method_vector(matrix, vector, eps)


def find_max_eigen_value_scalar(matrix, eps):
    dim = matrix.shape[0]
    vector = np.ones(dim)
    return exp_scalar_method(matrix, vector, eps)
