import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import preprocessor as pr
import newton as nw
import lagrange as lg
import spline as sl


def init_and_plot():
    x = sp.symbols('x')
    expr = 3 ** x
    a = -1
    b = 1
    n = 2
    strategy = 'E'
    plot(x, expr, a, b, n, strategy)


def add_to_plot(x, expr, a, b, name):
    if isinstance(expr, sp.Expr) and not expr.free_symbols:
        const_value = float(expr.evalf())
        x_values = np.linspace(a, b, 1000)
        y_values = np.full(1000, const_value)
    else:
        numpy_function = sp.lambdify(x, expr, 'numpy')
        x_values = np.linspace(a, b, 1000)
        y_values = numpy_function(x_values)
    plt.plot(x_values, y_values, label=name)


def add_points(vals):
    for val in vals:
        plt.scatter(val.argument(), val.function(), color='red', marker='o')


def init_plot():
    plt.figure(figsize=(8, 6))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Plot of the Function f(x)')


def plot_spline(spline_map):
    num_points = 1000
    x = sp.symbols('x')
    f = False
    for (x1, x2), expr in spline_map.items():
        numpy_function = sp.lambdify(x, expr, 'numpy')
        x_range = np.linspace(x1, x2, num_points)
        y_values = numpy_function(x_range)
        if f:
            plt.plot(x_range, y_values, color='violet')
        else:
            f = True
            plt.plot(x_range, y_values, color='violet', label='Spline')


def plot(x, expr, a, b, n, strategy='Optimal'):
    if n < 1:
        raise ValueError("Not enough points!")
    if strategy == 'Optimal' or strategy == 'O':
        chosen_vals = pr.optimal_values(expr, a, b, n)
    elif strategy == 'Even' or strategy == 'E':
        chosen_vals = pr.even_values(expr, a, b, n)
    else:
        raise ValueError('Unknown strategy!')

    init_plot()

    add_to_plot(x, expr, a, b, 'Initial function')
    newton = nw.newton_interpolation(chosen_vals)
    add_to_plot(x, newton, a, b, 'Newton')
    lagrange = lg.lagrange_interpolation(chosen_vals)
    add_to_plot(x, lagrange, a, b, 'Lagrange')
    spline = sl.spline_interpolation(chosen_vals)
    plot_spline(spline)

    add_points(chosen_vals)
    plt.grid(True)
    plt.legend()
    plt.savefig('lab5/result/plot.png', dpi=300, bbox_inches='tight')


init_and_plot()
