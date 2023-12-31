import numpy as np


def get_permutation_matrix(n, i, j):
    identity_matrix = np.eye(n)
    permutation_matrix = np.copy(identity_matrix)
    permutation_matrix[[i, j]] = permutation_matrix[[j, i]]
    return permutation_matrix


def get_transformation_matrix(matrix, n, i):
    result_matrix = np.eye(n)
    my_column = np.zeros(i)
    diag = 1.0 / matrix[i, i]
    my_column = np.append(my_column, diag)
    for j in range(i + 1, n):
        a = - matrix[j, i] * diag
        my_column = np.append(my_column, a)
    result_matrix[:, i] = my_column
    return result_matrix


def gaussian(matrix, b_vector, eps, do_test):
    assert matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]
    applied_matrices = []
    for i in range(n):
        sub_column = matrix[i:, i]
        max_index = i + np.argmax(sub_column)

        if do_test and np.abs(matrix[i][max_index]) < eps:
            raise TypeError("Matrix is singular")
        p = get_permutation_matrix(n, i, max_index)
        matrix = np.dot(p, matrix)
        m = get_transformation_matrix(matrix, n, i)
        matrix = np.dot(m, matrix)
        applied_matrices.append(p)
        applied_matrices.append(m)
    for to_apply in applied_matrices:
        b_vector = np.dot(to_apply, b_vector)
    x_vector = np.zeros(n)
    for i in range(n - 1, -1, -1):
        el_sum = 0
        for j in range(i+1, n):
            el_sum += matrix[i][j] * x_vector[j]
        x_vector[i] = b_vector[i] - el_sum
    return x_vector
