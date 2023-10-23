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
    do_test = config["TEST"]
    dim = config["DIMENSIONS"]
    pre_solution = gen.generate_solution(dim, 3)
    if config["GENERATE"]:
        matrix = gen.generate_hilbert_matrix(dim)
        vector = gen.from_solution(matrix, pre_solution)
    else:
        matrix, vector = par.read()
    if config["PRINT"]:
        print_mtx(matrix, vector)

    dim = matrix.shape[0]
    x_begin = np.zeros(dim)

    eps = config["EPSILON"]
    method = config["METHOD"]
    if method == "G":
        solution = gau.gaussian(matrix, vector, eps, do_test)
    elif method == "J":
        if do_test:
            jac.test_jacobi(matrix, eps)
        solution = jac.jacobi(matrix, vector, x_begin, eps)
    elif method == "S":
        if do_test:
            sei.test_seidel(matrix, eps)
        solution = sei.seidel(matrix, vector, x_begin, eps)
    else:
        raise "Unknown method"
    print("SOLUTION:")
    print(solution)
    if config["GENERATE"]:
        test.is_solution(matrix, vector, solution)
        test.compare(pre_solution, solution)


if __name__ == "__main__":
    config_map = {
        "GENERATE": True,
        "DIMENSIONS": 50,
        "PRINT": False,
        "TYPE": 4,
        "LIMIT": 100,
        "METHOD": "G",
        "EPSILON": 1e-5,
        "TEST": False,
    }
    calculate(config_map)
