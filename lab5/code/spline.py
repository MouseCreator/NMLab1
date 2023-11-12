import sympy as sp
import numpy as np


def diagonal_element(vals, i):
    return (get_h(vals, i) + get_h(vals, i + 1)) / 3


def semi_diagonal_element(vals, i):
    return (vals[i].argument()) / 6


def define_a(vals):
    m = len(vals) - 2
    a_matrix = np.zeros((m, m))
    for i in range(m):
        if i > 0:
            a_matrix[i][i - 1] = semi_diagonal_element(vals, i)
        a_matrix[i][i] = diagonal_element(vals, i)
        if i < m - 1:
            a_matrix[i][i + 1] = semi_diagonal_element(vals, i + 1)
    return a_matrix


def define_f(vals):
    f = []
    for v in vals:
        f.append(v.function())
    return f


def get_h(vals, i):
    return vals[i].argument() - vals[i - 1].argument()


def define_h(vals):
    m = len(vals) - 2
    h_matrix = np.zeros((m, m + 2))
    for i in range(m):
        h_matrix[i][i] = 1 / get_h(vals, i)
        h_matrix[i][i + 1] = -1 / get_h(vals, i) - 1 / get_h(vals, i + 1)
        h_matrix[i][i + 2] = 1 / get_h(vals, i + 1)
    return h_matrix


def solve(a_matrix, b_vector):
    return np.linalg.solve(a_matrix, b_vector)


def define_x(vals):
    x = []
    for v in vals:
        x.append(v.argument())
    return x


def spline_interpolation(vals):
    x = sp.symbols('x')
    a_matrix = define_a(vals)
    h_matrix = define_h(vals)
    f_vector = define_f(vals)
    x_vector = define_x(vals)
    b_vector = h_matrix.dot(f_vector).astype(np.float64)
    m_vector = solve(a_matrix, b_vector)
    splines = {}
    m_full = [0, m_vector, 0]
    for i in range(1, len(m_full)):
        h = get_h(vals, i)
        s = (m_full[i - 1] * (x_vector[i] - x) ** 3 / (6 * h) +
             + m_full[i] * (x - x_vector[i - 1]) ** 3 / (6 * h) +
             + (f_vector[i - 1] - m_full[i - 1] * h ** 2 / 6) * (x_vector[i] - x) / h +
             + (f_vector[i] - m_full[i] * h ** 2 / 6) * (x - x_vector[i - 1]) / h)
        splines[(x_vector[i - 1], x_vector[i])] = s[0]
    return splines
