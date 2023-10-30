import numpy as np
import sympy as sp
from newton import find_jacobian

def subs_to_all(functions, x_prev):
    n = len(functions)
    result = np.zeros(n)
    for i in range(n):
        result[i] = functions[i].subs(x_prev)
    return result


def estimate_tau(functions, variables, x_init):
    # Find tau?
    jacobian = find_jacobian(functions, variables)
    return 2 / np.linalg.norm(np.array(jacobian.subs(x_init)).astype(float), ord=1)


def relaxation(functions, variables, x_init, eps):
    tau = estimate_tau(functions, variables, x_init)
    x_prev = x_init
    while True:
        x_curr = x_prev - tau * subs_to_all(functions, x_prev)
        if np.abs(x_curr - x_prev) < eps:
            break
        x_prev = x_curr.copy()
    return x_curr