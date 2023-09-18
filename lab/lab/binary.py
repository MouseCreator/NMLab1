import math

import sympy as sp
import numpy as np

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

    return (a + b) / 2


def derivative_min(function, variable, a, b):
    xs = sp.symbols(variable)
    derivative = sp.diff(function, xs)

    derivative_func = sp.lambdify(xs, derivative, 'numpy')

    def to_min(x_value):
        return abs(derivative_func(x_value))

    result = minimize_scalar(to_min, bounds=(float(a), float(b)), method='bounded')

    return result.fun


def derivative_max(function, variable, a, b):
    xs = sp.symbols(variable)
    derivative = sp.diff(function, xs)

    derivative_func = sp.lambdify(xs, derivative, 'numpy')

    def to_min(x_value):
        return -abs(derivative_func(x_value))

    result = minimize_scalar(to_min, bounds=(float(a), float(b)), method='bounded')

    return -result.fun


def derivative_sign(function, variable, a, b):
    xs = sp.symbols(variable)
    derivative = sp.diff(function, xs)

    derivative_func = sp.lambdify(xs, derivative, 'numpy')

    def to_min(x_value):
        return (derivative_func(x_value))

    def to_max(x_value):
        return -(derivative_func(x_value))

    s1 = minimize_scalar(to_min, bounds=(float(a), float(b)), method='bounded')
    s2 = minimize_scalar(to_max, bounds=(float(a), float(b)), method='bounded')
    s2f = -s2.fun
    return s1.fun * s2f > 0


def iteration_apriory(function, a, b, tau, q, epsilon):
    z = b - a

    num_iterations = math.floor(math.log(abs(z) / epsilon) / math.log(1 / q)) + 1

    x = a
    xs = sp.symbols('x')

    for _ in range(num_iterations):
        x = x + tau * function.subs(xs, x)
    return x


def iteration_iter(function, a, b, tau, q, epsilon):
    x = a
    xn = b
    xs = sp.symbols('x')

    derivative = sp.diff(function, x)

    while abs(xn - x) > epsilon:
        x = xn
        xn = x + sp.sign(derivative.subs(x, xs)) * tau * function.subs(xs, x)
    return x


def iteration(function, a, b, epsilon, a_priory=False):
    q = 0
    tau = 0
    xs = sp.symbols('x')
    while True:
        deriv_min = derivative_min(function, 'x', a, b)
        deriv_max = derivative_max(function, 'x', a, b)

        tau = 2 / (deriv_min + deriv_max)
        q = (deriv_max - deriv_min) / (deriv_max + deriv_min)
        if q < 0.99 and derivative_sign(function, 'x', a, b):
            break
        x = (a + b) / 2
        val = function.subs(xs, x)
        if sp.sign(function.subs(xs, a)) == sp.sign(val):
            a = x
        if sp.sign(function.subs(xs, b)) == sp.sign(val):
            b = x

    if sp.diff(function, xs).subs(xs, a) > 0:
        tau *= -1

    if a_priory:
        return iteration_apriory(function, a, b, tau, q, epsilon)
    else:
        return iteration_iter(function, a, b, tau, q, epsilon)


def newton(function, a, b, epsilon, a_priory=False):
    q = 2
    xs = 'x'
    sym = sp.symbols(xs)
    while True:
        deriv = sp.diff(function, sym)
        q = derivative_max(deriv, xs, a, b) * (b - a) / (2 * derivative_min(function, xs, a, b))
        x = (a + b) / 2
        val = function.subs(xs, x)
        if q < 1:
            break
        if sp.sign(function.subs(xs, a)) == sp.sign(val):
            a = x
        if sp.sign(function.subs(xs, b)) == sp.sign(val):
            b = x

    x = a
    x0 = b

    while abs(x0 - x) > epsilon:
        x0 = x
        x = x - function.subs(xs, x) / deriv.subs(xs, x)
    return x
