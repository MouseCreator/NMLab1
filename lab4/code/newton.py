import numpy as np
import sympy as sp


def find_Jacobian(functions, variables):
    return sp.Matrix([[sp.diff(f, var) for var in variables] for f in functions])


def newton_calc(functions, jacobian, eps):
    x_init = np.ones(functions.length)
    x_curr = x_init
    while True:
        a_matrix = jacobian.subs(x_curr)
        b_vector = functions.subs(x_curr)
        z = np.linalg.solve(a_matrix, b_vector)
        x_curr = x_curr - z
        if np.linalg.norm(z) < eps:
            break
    return x_curr


def newton(functions, variables, eps):
    jacobian = find_Jacobian(functions, variables)
    newton_calc(functions, jacobian, eps)
