import numpy as np
import sympy as sp
from newton import find_jacobian, create_function, initial_solution
from scipy.optimize import minimize


def subs_to_all(functions, x_prev):
    n = len(functions)
    result = np.zeros(n)
    for i in range(n):
        result[i] = functions[i].subs(x_prev)
    return result


def to_regular_vector(x_init):
    return np.array(list(x_init.values()))


def to_dictionary(x_init, x):
    x_p = x_init.copy()
    if len(x) == len(x_p):
        for key, new_value in zip(x_p.keys(), x):
            x_p[key] = new_value
        return x_p
    else:
        raise ValueError("The length of new_values does not match the number of keys in the dictionary.")


def generate_bounds(x_j, delta):
    bounds = []
    for x_i in x_j:
        bounds.append([x_i - delta, x_i + delta])
    return np.array(bounds)


def estimate_tau(functions, variables, x_init, delta=10):
    delta_vector = np.full(len(x_init.values()), delta)
    x_j = to_regular_vector(x_init)

    jacobian = find_jacobian(functions, variables)
    bounds = generate_bounds(x_j, delta)

    def neg_jacobian_norm(x):
        y = to_dictionary(x_init, x)
        jm = np.array(jacobian.subs(y)).astype(float)
        return -np.linalg.norm(jm)

    result = minimize(neg_jacobian_norm, x0=x_j, bounds=bounds)

    return -2.0 / result.fun


def update_relaxation_solution(x_prev, tau, functions):
    updated = x_prev.copy()
    fx = tau * subs_to_all(functions, x_prev)
    i = 0
    for v in x_prev:
        updated[v] -= fx[i]
        i += 1
    return updated


def measure_difference(x1, x2):
    sq_sum = 0
    for v in x1:
        sq_sum += (x1[v] - x2[v]) ** 2
    return np.sqrt(sq_sum)


def relaxation(functions, variables, x_init, eps):
    tau = estimate_tau(functions, variables, x_init)
    x_prev = x_init
    while True:
        x_curr = update_relaxation_solution(x_prev, tau, functions)
        if measure_difference(x_curr, x_prev) < eps:
            break
        x_prev = x_curr.copy()
    return x_curr


def test_relaxation_n_space(n, eps=1e-6):
    variables = [sp.symbols(f'x_{i}') for i in range(1, n + 1)]
    functions = []
    init = initial_solution(variables, 1)
    for i in range(n):
        f = create_function(variables, n, i)
        functions.append(f)
    solution = relaxation(functions, variables, init, eps)
    print(solution)


def test_relaxation(eps=1e-6):
    x, y = sp.symbols('x y')
    f1 = x ** 2 - 2 * x * y + 1
    f2 = x ** 2 + y ** 2 - 2
    funcs = [f1, f2]
    variables = [x, y]
    init = initial_solution(variables, 0.75)
    solution = relaxation(funcs, variables, init, eps)
    print(solution)
