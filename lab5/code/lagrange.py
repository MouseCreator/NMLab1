import val as v
import sympy as sp


def at(expression, x_value):
    x = sp.symbols('x')
    x_map_t = {x: x_value}
    return expression.subs(x_map_t)


def simplify(expr):
    return sp.simplify(expr)


def lagrange_interpolation(vals):
    x = sp.symbols('x')
    v_prod = v_product(vals, x)
    v_der = derivative_v_product(v_prod, x)
    polynomial = 0
    for val in vals:
        polynomial = polynomial + (v_prod * val.function()) / ((x - val.argument()) * at(v_der, val.argument()))
    return simplify(polynomial)


def v_product(vals, x):
    if not vals:
        return 1
    else:
        expression = 1
        for val in vals:
            expression *= (x - val.argument())
        return expression


def derivative_v_product(expression, x):
    return sp.diff(expression, x)


v1 = v.Val([-1, 1 / 3])
v2 = v.Val([0, 1])
v3 = v.Val([1, 3])

p = lagrange_interpolation([v1, v2, v3])
print(p)
