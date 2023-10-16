import numpy as np

import lab2.code.generator as gen
import lab2.code.gaussian as gau
import lab2.code.parser as par
import lab2.code.jacobi as jac
import lab2.code.seidel as sei
import lab2.code.test as test


def print_mtx(matrix, vector):
    for row, v in zip(matrix, vector):
        print("\t".join(map(str, row)), "|", v)

def print_mtx_s(matrix, n):
    for i in range(n):
        s = ""
        for j in range(n):
            s += str(matrix[i][j])
            s += " "
        print(s)


def calculate(config):
    limit = config["LIMIT"]
    dim = config["DIMENSIONS"]
    if config["GENERATE"]:
        matrix = gen.generate_random_matrix_eigen_values_based(dim)
        vector = gen.generate_vector(dim, 1.0)
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
    test.is_solution(matrix, vector, solution)


# TODO: add autotest
# TODO: fix eigen values generator
# TODO: (1,1,1) => calculate b => solve. Expect: (1, 1, 1). ||Actual - expected|| -> 0
if __name__ == "__main__":
    config_map = {
        "GENERATE": True,
        "DIMENSIONS": 5,
        "PRINT": True,
        "TYPE": 4,
        "LIMIT": 100,
        "METHOD": "S",
        "EPSILON": 0.0001
    }
    #print_mtx_s(gen.generate_random_matrix_eigen_values_based(10), 10)
    calculate(config_map)

