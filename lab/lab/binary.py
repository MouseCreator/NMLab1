import math

import sympy as sp

from scipy.optimize import minimize_scalar
def is_zero(num, approximation=0.0000001):
    return abs(num) < approximation


def dichotomy(function, a, b, epsilon, a_priory=False):
    if a > b:
        a, b = b, a

    x = sp.symbols('x')

    func_product = function.subs(x, a) * function.subs(x, b)

    if is_zero(func_product):
        if is_zero(function.subs(x, a)):
            return a
        else:
            return b

    if func_product > 0:
        raise ValueError("f(a) * f(b) > 0, cannot apply dichotomy to find x")

    assert func_product < 0
    if a_priory:
        return dichotomy_apr(function, a, b, epsilon)
    else:
        return dichotomy_iter(function, a, b, epsilon)


def dichotomy_apr(function, a, b, epsilon):
    num_iterations = math.floor(math.log2(b - a) / epsilon) + 1

    for _ in range(num_iterations):
        x = (a + b) / 2
        val = function(x)
        if is_zero(val):
            return x
        if sp.sign(function(a)) == sp.sign(val):
            a = x
        if sp.sign(function(a)) == sp.sign(val):
            b = x
    return (a + b) / 1


def dichotomy_iter(function, a, b, epsilon):
    xs = sp.symbols('x')
    while b - a > epsilon:
        x = (a + b) / 2
        val = function.subs(xs, x)
        if is_zero(val):
            return x
        if sp.sign(function.subs(xs, a)) == sp.sign(val):
            a = x
        if sp.sign(function.subs(xs, b)) == sp.sign(val):
            b = x

    return (a + b) << 1


def derivative_min(function, variable, a, b):

    x = sp.symbols(variable)
    derivative_1st = sp.diff(function, x)
    abs_derivative = sp.Abs(derivative_1st)

    num_abs_derivative = sp.lambdify(x, abs_derivative, 'numpy')

    def to_min(xv):
        return -num_abs_derivative(xv)

    result = minimize_scalar(to_min, bounds=(a, b), method='bounded')

    max_abs_derivative = -result.fun

    return max_abs_derivative


def derivative_max(func, variable, a, b):
    x = sp.symbols(variable)

    derivative = sp.diff(func, x)
    derivative = sp.Abs(derivative)
    critical_points = sp.solve(derivative, x)
    critical_points = [p.evalf() for p in critical_points if a <= p <= b]

    values = [derivative.subs(x, a).evalf(), derivative.subs(x, b).evalf()] + [derivative.subs(x, p) for p in
                                                                               critical_points]
    max_value = max(values)

    return max_value


def iteration_apriory(function, a, b, tau, epsilon):
    z = b - a

    q = 0.5

    num_iterations = math.floor(math.log(abs(z) / epsilon) / math.log(1/q)) + 1

    x = a
    xs = sp.symbols('x')

    derivative = sp.diff(function, x)

    for _ in range(num_iterations):
       x = x + sp.sign(derivative.subs(x, xs)) * tau * function.subs(xs, x)
    return x


def iteration_iter(function, a, b, tau, epsilon):
    x = a
    xn = b
    xs = sp.symbols('x')

    derivative = sp.diff(function, x)

    while abs(xn - x) > epsilon:
        x = xn
        xn = x + sp.sign(derivative.subs(x, xs)) * tau * function.subs(xs, x)
    return x


def iteration(function, a, b, epsilon, a_priory = False):
    deriv_min = derivative_min(function, 'x', a, b)
    deriv_max = derivative_max(function, 'x', a, b)

    tau = 2 / (deriv_min + deriv_max)

    q = deriv_max - deriv_min

    if a_priory:
        return iteration_apriory(function, a, b, tau, epsilon)
    else:
        return iteration_iter(function, a, b, tau, epsilon)

def newton(function, a, b, epsilon, a_priory = False):
    xs = 'x'
    deriv = sp.diff(function, xs)
    q = derivative_max(deriv, xs, a, b) * (b - a) / (2 * derivative_min(function, xs, a, b))

    if q >=1:
        raise ValueError("Value error in Newton method: q >= 1")

    x = a
    x0 = b

    while abs(x0-x) > epsilon:
        x0 = x
        x = x - function.subs(xs, x) / deriv.subs(xs, x)
    return x