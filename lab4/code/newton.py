import numpy as np
import sympy as sp


def find_Jacobian(functions, variables):
    return sp.Matrix([[sp.diff(f, var) for var in variables] for f in functions])


def calculate_function(functions, x_curr):
    n = len(functions)
    results = np.zeros(n)
    for i in range(n):
        func = functions[i]
        result = func.subs(x_curr)
        results[i] = result
    return results


def initial_solution(variables):
    solution_map = {}
    for v in variables:
        solution_map[v] = 0.25
    return solution_map


def update_solution(x_curr, z):
    result = x_curr
    i = 0
    for key in x_curr:
        result[key] -= z[i]
        i += 1
    return result



def newton_calc(functions, variables, jacobian, eps):
    x_init = initial_solution(variables)
    x_curr = x_init
    while True:
        a_matrix = np.array(jacobian.subs(x_curr)).astype(float)
        b_vector = calculate_function(functions, x_curr)
        z = np.linalg.solve(a_matrix, b_vector)
        x_curr = update_solution(x_curr, z)
        if np.linalg.norm(z) < eps:
            break
    return x_curr


def newton(functions, variables, eps):
    jacobian = find_Jacobian(functions, variables)
    return newton_calc(functions, variables, jacobian, eps)


def test_newton():
    x, y = sp.symbols('x y')
    f1 = x ** 2 - 2 * x * y + 1
    f2 = x ** 2 + y ** 2 - 2
    funcs = [f1, f2]
    vars = [x, y]
    eps = 1e-5
    solution = newton(funcs, vars, eps)
    print(solution)


test_newton()
