import numpy as np


def generate_random_matrix(dim):
    return np.random.rand(dim, dim)


def generate_random_int_matrix(dim, lim=101):
    return np.random.randint(low=0, high=lim, size=(dim, dim))


def generate_random_matrix_diagonal_dominance(dim, lim=101):
    matrix = np.random.rand(dim, dim) * lim
    diagonal_elements = np.sum(np.abs(matrix), axis=1) + np.random.rand(dim) * lim
    np.fill_diagonal(matrix, diagonal_elements)

    return matrix


def generate_random_matrix_significant_diagonal_dominance(dim, lim=101):
    matrix = np.random.rand(dim, dim) * lim
    diagonal_elements = 1.7 * np.sum(np.abs(matrix), axis=1) + (np.random.rand(dim) * lim) * 1.5
    np.fill_diagonal(matrix, diagonal_elements)

    return matrix


def generate_orthogonal_matrix(dim):
    random_matrix = np.random.randn(dim, dim)
    q, r = np.linalg.qr(random_matrix)
    return q


def generate_non_singular(dim, limit):
    while True:
        matrix = generate_random_matrix(dim)
        if np.abs(np.linalg.det(matrix)) > 0.001:
            return matrix * limit


def generate_random_matrix_eigen_values_based(dim):
    eigenvalues_modulo = np.random.uniform(low=0.01, high=0.9, size=dim)
    signs = np.random.choice([-1, 1], size=dim)
    eigenvalues = eigenvalues_modulo * signs

    diagonal_matrix = np.diag(eigenvalues)
    orthogonal_matrix = generate_orthogonal_matrix(dim)
    b = np.dot(np.dot(orthogonal_matrix, diagonal_matrix), orthogonal_matrix.T)

    rand_diag = np.random.uniform(low=0.01, high=0.9, size=dim)
    d = np.diag(rand_diag)
    return np.dot(d, b) + d


def generate_hilbert_matrix(dim):
    h = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            h[i][j] = 1 / (i + j + 1)

    return h


def generate_vector(dim, lim=101):
    return np.random.uniform(low=-lim, high=lim, size=dim)


def generate_int_vector(dim, lim=101):
    return np.random.randint(low=-lim, high=lim, size=dim)


def generate_solution(dim, fill):
    return np.full(dim, fill)


def from_solution(matrix, solution):
    return np.dot(matrix, solution)
