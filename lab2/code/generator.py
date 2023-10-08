import numpy as np


def generate_random_matrix(dim):
    return np.random.rand(dim, dim)


def generate_random_int_matrix(dim):
    return np.random.randint(low=0, high=101, size=(dim, dim))


def generate_random_matrix_diagonal_priority(dim):
    return np.random.randint(low=0, high=101, size=(dim, dim))


def generate_orthogonal_matrix(dim):
    random_matrix = np.random.randn(dim, dim)
    q, r = np.linalg.qr(random_matrix, mode='reduced')
    return q


def generate_random_matrix_eigen_values_based(dim):
    eigenvalues_modulo = np.random.uniform(low=0.01, high=0.9, size=dim)
    signs = np.random.choice([-1, 1], size=dim)
    eigenvalues = eigenvalues_modulo * signs

    diagonal_matrix = np.diag(eigenvalues)
    orthogonal_matrix = generate_orthogonal_matrix(dim)
    return np.dot(np.dot(orthogonal_matrix, diagonal_matrix), orthogonal_matrix.T)


def generate_hilbert_matrix(dim):
    h = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            h[i][j] = 1/(i+j+1)

    return h

