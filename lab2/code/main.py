import numpy as np

import lab2.code.generator as gen
import lab2.code.gaussian as gau
import lab2.code.parser as par
import lab2.code.jacobi as jac
import lab2.code.seidel as sei


def print_mtx(matrix, vector):
    for row, v in zip(matrix, vector):
        print("\t".join(map(str, row)), "|", v)


def calculate(config):
    limit = config["LIMIT"]
    dim = config["DIMENSIONS"]
    if config["GENERATE"]:
        matrix = gen.generate_random_matrix_diagonal_dominance(dim, limit)
        vector = gen.generate_int_vector(dim, limit)
    else:
        matrix, vector = par.read()
    if config["PRINT"]:
        print_mtx(matrix, vector)

    dim = matrix.shape[0]
    x_begin = np.zeros(dim)

    eps = config["EPSILON"]
    method = config["METHOD"]
    if method == "G":
        solution = gau.gaussian(matrix, vector, eps)
    elif method == "J":
        solution = jac.jacobi(matrix, vector, x_begin, eps)
    elif method == "S":
        solution = sei.seidel(matrix, vector, x_begin, eps)
    else:
        raise "Unknown method"
    print("SOLUTION:")
    print(solution)


if __name__ == "__main__":
    config_map = {
        "GENERATE": False,
        "DIMENSIONS": 10,
        "PRINT": True,
        "TYPE": 4,
        "LIMIT": 100,
        "METHOD": "J",
        "EPSILON": 0.0001
    }
    calculate(config_map)
