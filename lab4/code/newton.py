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


def initial_solution(variables, val=0.0):
    solution_map = {}
    for v in variables:
        solution_map[v] = val
    return solution_map


def update_solution(x_curr, z):
    result = x_curr
    i = 0
    for key in x_curr:
        result[key] -= z[i]
        i += 1
    return result


def newton_calc(functions, variables, jacobian, eps):
    x_init = initial_solution(variables, 0.25)
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
def max_second_derivative(functions, x_init):
    max_val = 0
    for f in functions:
        for v in x_init:
            f_prime = sp.diff(f, v)
            f_double_prime = sp.diff(f_prime, v)
            mx = np.abs(f_double_prime.subs(x_init))
            if mx > max_val:
                max_val = mx
    return max_val


def max_f(functions, x_init):
    max_val = 0
    for f in functions:
        mx = np.abs(f.subs(x_init))
        if mx > max_val:
            max_val = mx
    return max_val


def test_solvable(functions, jacobian, x_init, eps):
    A = np.array(jacobian.subs(x_init)).astype(float)
    n = len(functions)
    if np.abs(np.linalg.det(A)) < eps:
        return False
    L = max_second_derivative(functions, x_init)
    M = np.linalg.norm(np.linalg.inv(A), ord=2)
    D = max_f(functions, x_init)
    return np.abs(M * M * L * D * n * n - 1/2) < eps


def create_function(variables, n, i):
    f = -n
    j = 0
    for v in variables:
        if j == i:
            f += v ** 3
        else:
            f += v ** 2
        j += 1
    return f


def test_newton_n_space(n, eps):
    variables = [sp.symbols(f'x_{i}') for i in range(1, n + 1)]
    functions = []
    for i in range(n):
        f = create_function(variables, n, i)
        functions.append(f)
    solution = newton(functions, variables, eps)
    print(solution)

def test_newton(eps):
    x, y = sp.symbols('x y')
    f1 = x ** 2 - 2 * x * y + 1
    f2 = x ** 2 + y ** 2 - 2
    funcs = [f1, f2]
    vars = [x, y]
    solution = newton(funcs, vars, eps)
    print(solution)

def is_solvable():
    x, y = sp.symbols('x y')
    f1 = x - 0.5 * sp.sin((x-y)/2)
    f2 = y - 0.5 * sp.cos((x+y)/2)
    funcs = [f1, f2]
    vars = [x, y]
    J = find_Jacobian(funcs, vars)
    x_init = initial_solution(vars, 0)
    print("Solvable?", test_solvable(funcs, J, x_init, 1e-5))


is_solvable()

