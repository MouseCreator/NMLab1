import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import preprocessor as pr
import newton as nw
import lagrange as lg


def init_and_plot():
    x = sp.symbols('x')
    expr = 3 ** x
    a = -1
    b = 1
    n = 3
    plot(x, expr, a, b, n)


def add_to_plot(x, expr, a, b, name):
    numpy_function = sp.lambdify(x, expr, 'numpy')
    x_values = np.linspace(a, b, 100)
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


def plot(x, expr, a, b, n):
    optimal_values = pr.optimal_values(expr, a, b, n)

    init_plot()

    add_to_plot(x, expr, a, b, 'Initial function')
    newton = nw.newton_interpolation(optimal_values)
    add_to_plot(x, newton, a, b, 'Newton')
    lagrange = lg.lagrange_interpolation(optimal_values)
    add_to_plot(x, lagrange, a, b, 'Lagrange')

    add_points(optimal_values)
    plt.grid(True)
    plt.legend()
    plt.show()


init_and_plot()
