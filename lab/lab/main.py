import sympy as sp
import lab.lab.binary as di


def execute_task(task):
    if task["method"] == "dichotomy":
        execute_dichotomy(task)
    elif task["method"] == "iteration":
        execute_dichotomy(task)
    elif task["method"] == "newton":
        execute_dichotomy(task)


def execute_dichotomy(task):
    function = sp.parse_expr(task["equation"])
    epsilon = sp.parse_expr(task["epsilon"])
    a = sp.parse_expr(task["a"])
    b = sp.parse_expr(task["b"])
    a_priory = sp.parse_expr(task["a_priory"])

    root = di.dichotomy(function, a, b, epsilon, a_priory)
    print(task["name"] + " found root of " + task["equation"] + ": " + root)
