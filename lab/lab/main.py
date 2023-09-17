import sympy as sp
import lab.lab.binary as di
import lab.lab.io as io


def execute_task(task):
    if task["method"] == "dichotomy":
        execute_dichotomy(task)
    elif task["method"] == "iteration":
        execute_dichotomy(task)
    elif task["method"] == "newton":
        execute_dichotomy(task)


def to_bool(str):
    if str.lower() == "true":
        return True
    else:
        return False
def execute_dichotomy(task):
    function = sp.parse_expr(task["equation"])
    epsilon = sp.parse_expr(task["epsilon"]).evalf()
    a = sp.parse_expr(task["a"]).evalf()
    b = sp.parse_expr(task["b"]).evalf()
    a_priory = to_bool(task["a_priory"])
    try:
        root = di.dichotomy(function, a, b, epsilon, a_priory)
        formatted_string = "{} found root of {}:\n{}={}".format(task["name"], task["equation"], task["variable"], root)
        print(formatted_string)
    except Exception as e:
        formatted_string = "{}: ERROR!\n{}".format( task["name"], e)
        print(formatted_string)



if __name__ == "__main__":
    filename = "../input\\input.txt"
    try:
        tasks = io.parse_task_file(filename)
    except Exception as e:
        print(f"An error occurred: {e}")

    if tasks:
        for t in tasks:
            execute_task(t)
