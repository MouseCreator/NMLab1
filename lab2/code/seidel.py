import numpy as np

from lab2.code.jacobi import to_non_zero_diagonal


def seidel(matrix, b, x0, eps):
    matrix = to_non_zero_diagonal(matrix, eps)
    n = matrix.shape[0]
    x_prev = x0
    x_cur = np.zeros(n)
    while True:
        for i in range(n):
            el_sum = 0
            for j in range(i):
                el_sum += matrix[i][j] * x_cur[j]
            for j in range(i+1, n):
                el_sum += matrix[i][j] * x_prev[j]
            x_cur[i] = (b[i] - el_sum) / matrix[i][i]
        difference = x_cur - x_prev
        difference_norm = np.linalg.norm(difference)
        if difference_norm < eps:
            return x_cur
        x_prev = x_cur.copy()

