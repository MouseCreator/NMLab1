import val as v
import sympy as sp
import numpy as np


def at(expression, x_value):
    x = sp.symbols('x')
    x_map_t = {x: x_value}
    return expression.subs(x_map_t)


def lagrange_interpolation(vals):
    x = sp.symbols('x')
    v_prod = v_product(vals, x)
    v_der = derivative_v_product(v_prod, x)
    polynomial = 0
    for val in vals:
        polynomial = polynomial + (v_prod * val.function()) / (x - val.argument() * at(v_der, val.argument()))
    return polynomial


def v_product(vals, x):
    if not vals:
        return 1
    else:
        expression = 1
        for val in vals:
            expression *= (x - val)
        return expression


def derivative_v_product(expression, x):
    return sp.diff(expression, x)
