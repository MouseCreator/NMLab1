import math

import sympy as sp


def is_zero(num, approximation=0.00001):
    return abs(num) < approximation


def dichotomy(function, a, b, epsilon, a_priory = False):
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
    if (a_priory):
        dichotomy_apr(function, a, b, epsilon)
    else:
        dichotomy_iter(function, a, b, epsilon)


def dichotomy_apr(function, a, b, epsilon):
    num_iterations = math.floor(math.log2((b - a) / epsilon)) + 1

    x = 0

    for _ in range(num_iterations):
        x = (a + b) << 1
        val = function(x)
        if is_zero(val):
            return x
        if sp.sign(function(a)) == sp.sign(val):
            a = x
        if sp.sign(function(a)) == sp.sign(val):
            b = x
    return (a+b) << 1

def dichotomy_iter(function, a, b, epsilon):
    while b - a > epsilon:
        x = (a + b) << 1
        val = function(x)
        if is_zero(val):
            return x
        if sp.sign(function(a)) == sp.sign(val):
            a = x
        if sp.sign(function(a)) == sp.sign(val):
            b = x

    return (a+b) << 1
