import numpy as np


def swap_rows(matrix, row1, row2):
    matrix[[row1, row2]] = matrix[[row2, row1]]
    return matrix


def test_eigen_values(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(np.abs(eigenvalues) < 1)


def find_and_swap(matrix, target, eps):
    n = matrix.shape[0]
    for j in range(n):
        if j == target:
            continue
        if np.abs(matrix[target][j]) > eps and np.abs(matrix[j][target] > eps):
            return swap_rows(matrix, j, target)
    raise TypeError("Unable to create a matrix with non-zero diagonal")


def to_non_zero_diagonal(matrix, eps):
    diagonal_elements = np.diag(matrix)
    n = diagonal_elements.shape[0]
    for i in range(n):
        if np.abs(diagonal_elements[i]) < eps:
            matrix = find_and_swap(matrix, i, eps)

    return matrix


def find_q(matrix, eps):
    n = matrix.shape[0]
    q = 0
    for i in range(n):
        diag = np.abs(matrix[i][i])
        el_sum = 0
        for j in range(n):
            if i == j:
                continue
            el_sum += np.abs(matrix[i][j])
        if diag < el_sum:
            return 1
        if diag - eps > el_sum:
            q = max(q, el_sum / diag)
    if q == 0:
        q = 1
    return q


def is_diagonally_dominant(matrix, eps):
    n = matrix.shape[0]
    has_strict = False
    for i in range(n):
        diag = np.abs(matrix[i][i])
        el_sum = 0
        for j in range(n):
            if i == j:
                continue
            el_sum += np.abs(matrix[i][j])
        if diag < el_sum:
            return False
        if diag - eps > el_sum:
            has_strict = True
    return has_strict


def test_jacobi(matrix, eps):
    diag = is_diagonally_dominant(matrix, eps)
    if not diag:
        raise ValueError("Matrix is not diagonally dominant!")


def jacobi(matrix, b, x0, eps):
    matrix = to_non_zero_diagonal(matrix, eps)
    n = matrix.shape[0]
    x_prev = x0
    x_cur = np.zeros(n)
    iters = 0
    while True:
        iters += 1
        for i in range(n):
            el_sum = 0
            for j in range(n):
                if j == i:
                    continue
                el_sum += matrix[i][j] * x_prev[j]
                x_cur[i] = (b[i] - el_sum) / matrix[i][i]
        difference = x_cur - x_prev
        difference_norm = np.linalg.norm(difference)
        if difference_norm < eps:
            print("ITERATIONS", iters)
            return x_cur
        x_prev = x_cur.copy()
