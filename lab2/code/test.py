import numpy as np


def is_solution(matrix, b, x):
    result = np.dot(matrix, x)

    difference = result - b
    difference_norm = np.linalg.norm(difference)
    print("Difference:", difference_norm)
