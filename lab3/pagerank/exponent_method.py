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


def exp_method_r(matrix, eps):
    b_k = np.random.rand(matrix.shape[1])
    while True:
        b_k1 = np.dot(matrix, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_prev = b_k
        b_k = b_k1 / b_k1_norm
        if np.linalg.norm(b_prev - b_k) < eps:
            break

    return b_k


def exp_method_equal_magnitude(matrix, eps):
    b_k = np.random.rand(matrix.shape[1])
    lbd = 0
    while True:
        b_k1 = np.dot(matrix, b_k)
        b_k1 = np.dot(matrix, b_k1)
        lbd_prev = lbd
        lbd = np.dot(b_k1, b_k) / np.dot(b_k, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        if np.abs(lbd - lbd_prev) < eps:
            break

    return np.sqrt(np.abs(lbd))


def to_eigenvector(matrix, eigenvalue):
    null_space = matrix - eigenvalue * np.eye(matrix.shape[0])

    _, _, v = np.linalg.svd(null_space)
    eigenvector = v[-1]

    return eigenvector


def exp_method_equal_magnitude_vector(matrix, eps):
    max_v = exp_method_equal_magnitude(matrix, eps)
    return to_eigenvector(matrix, max_v)


def exp_method_vector(matrix, vector, eps):
    if np.all(vector == 0):
        raise ValueError("Vector cannot be zero vector")
    while True:
        b_k1 = np.dot(matrix, vector)
        b_k1_norm = np.linalg.norm(b_k1)
        if (np.linalg.norm(b_k1 / b_k1_norm - vector)) < eps:
            return b_k1
        vector = b_k1 / b_k1_norm


def exp_method_vector_new(matrix, vector, eps):
    eigenvalue = exp_scalar_method(matrix, vector, eps)
    return to_eigenvector(matrix, eigenvalue)


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
    vector = np.random.rand(matrix.shape[1])
    return exp_method_vector_new(matrix, vector, eps)


def find_eigen_vector_equal_magnitude(matrix, eps):
    return exp_method_equal_magnitude_vector(matrix, eps)


def find_max_eigen_value_scalar(matrix, eps):
    dim = matrix.shape[0]
    vector = np.ones(dim)
    return exp_scalar_method(matrix, vector, eps)


def is_eigen_value(matrix, value, eps):
    det_1 = np.linalg.det(matrix - value * np.eye(matrix.shape[0]))
    return np.abs(det_1) < eps


def find_eigen_vector_auto(matrix, eps):
    if is_eigen_value(matrix, -1, eps):
        return find_eigen_vector_equal_magnitude(matrix, eps)
    else:
        return find_eigen_vector(matrix, eps)
