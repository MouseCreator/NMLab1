import numpy as np

import lab2.code.generator as gen
import lab2.code.gaussian as gau
import lab2.code.parser as par
import lab2.code.jacobi as jac
import lab2.code.seidel as sei


def print_mtx(matrix, vector):
    for row, v in zip(matrix, vector):
        print(" ".join(map(str, row)), "|", v)


def calculate(config):
    limit = config["LIMIT"]
    dim = config["DIMENSIONS"]
    if config["GENERATE"]:
        gen_type = config["TYPE"]
        if gen_type == 0:
            matrix = gen.generate_random_matrix(dim)
        elif gen_type == 1:
            matrix = gen.generate_random_int_matrix(dim, limit)
        elif gen_type == 2:
            matrix = gen.generate_random_matrix_diagonal_dominance(dim, limit)
        elif gen_type == 3:
            matrix = gen.generate_random_matrix_eigen_values_based(dim)
        elif gen_type == 4:
            matrix = gen.generate_hilbert_matrix(dim)
        elif gen_type == 5:
            matrix = gen.generate_non_singular(dim, limit)
        else:
            raise "Unknown type"
        if gen_type == 1:
            vector = gen.generate_int_vector(dim, limit)
        else:
            vector = gen.generate_vector(dim, limit)
        if config["PRINT"]:
            print_mtx(matrix, vector)
    else:
        matrix, vector = par.read()

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
        "GENERATE": True,
        "DIMENSIONS": 10,
        "PRINT": True,
        "TYPE": 1,
        "LIMIT": 100,
        "METHOD": "G",
        "EPSILON": 0.001
    }
    calculate(config_map)
