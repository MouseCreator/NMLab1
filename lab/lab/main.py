import sympy as sp
import lab.lab.binary as di
import lab.lab.io as io


def execute_task(task):
    method = task["method"].lower()
    if method == "dichotomy" or method == "d":
        execute_dichotomy(task)
    elif method == "iteration" or method == "i":
        execute_iter(task)
    elif method == "newton" or method == "n":
        execute_newton(task)
    else:
        print("Unknown method " + task["method"])


def to_bool(strval):
    if strval.lower() == "true":
        return True
    else:
        return False


def execute_dichotomy(task):
    try:
        expr = prep_expr(task["equation"])
        function = sp.parse_expr(expr)
        epsilon = sp.parse_expr(prep_expr_simple(task["epsilon"])).evalf()
        a = sp.parse_expr(prep_expr_simple(task["a"])).evalf()
        b = sp.parse_expr(prep_expr_simple(task["b"])).evalf()
        a_priory = to_bool(task["a_priory"])
        strict = to_bool(task["strict"])
        root = di.dichotomy(function, a, b, epsilon, a_priory, strict)
        formatted_string = "{} found root of {}:\n{}={}".format(task["name"], task["equation"], task["variable"], root)
        print(formatted_string)
    except Exception as e:
        formatted_string = "{}: ERROR!\n{}".format(task["name"], e)
        print(formatted_string)


def prep_expr(exp):
    exp = prep_expr_simple(exp)

    if exp.count("=") > 0:
        exp.replace("=", "-(")
        exp = exp + ")"
    return exp


def prep_expr_simple(str):
    str = str.replace("^", "**")
    return str


def execute_iter(task):
    try:
        expr = prep_expr(task["equation"])
        function = sp.parse_expr(expr)
        epsilon = sp.parse_expr(prep_expr_simple(task["epsilon"])).evalf()
        a = sp.parse_expr(prep_expr_simple(task["a"])).evalf()
        b = sp.parse_expr(prep_expr_simple(task["b"])).evalf()
        a_priory = to_bool(task["a_priory"])
        strict = to_bool(task["strict"])
        root = di.iteration(function, a, b, epsilon, a_priory, strict)
        formatted_string = "{} found root of {}:\n{}={}".format(task["name"], task["equation"], task["variable"], root)
        print(formatted_string)
    except Exception as e:
        formatted_string = "{}: ERROR!\n{}".format(task["name"], e)
        print(formatted_string)


def execute_newton(task):
    try:
        expr = prep_expr(task["equation"])
        function = sp.parse_expr(expr)
        epsilon = sp.parse_expr(prep_expr_simple(task["epsilon"])).evalf()
        a = sp.parse_expr(prep_expr_simple(task["a"])).evalf()
        b = sp.parse_expr(prep_expr_simple(task["b"])).evalf()
        a_priory = to_bool(task["a_priory"])
        strict = to_bool(task["strict"])
        root = di.newton(function, a, b, epsilon, a_priory, strict)
        formatted_string = "{} found root of {}:\n{}={}".format(task["name"], task["equation"], task["variable"], root)
        print(formatted_string)
    except Exception as e:
        formatted_string = "{}: ERROR!\n{}".format(task["name"], e)
        print(formatted_string)


if __name__ == "__main__":
    filename = "../input\\input.txt"
    try:
        tasks = io.parse_task_file(filename)
        for t in tasks:
            execute_task(t)
    except Exception as e:
        print(f"An error occurred: {e}")