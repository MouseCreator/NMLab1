import numpy as np


def parse_input(input_string):
    rows = input_string.strip().split('\n')

    matrix_a = []
    vector_b = []

    for row in rows:
        if not row.strip():
            continue
        if '|' in row:
            elements = row.strip().split('|')
            coefficients = list(map(float, elements[0].strip().split()))
            constant = float(elements[1].strip())
            matrix_a.append(coefficients)
            vector_b.append(constant)
        else:
            elements = list(map(float, row.strip().split()))
            n = len(elements) - 1
            matrix_a.append(elements[:n])
            vector_b.append(elements[n])
    matrix_a = np.array(matrix_a)
    vector_b = np.array(vector_b)

    return matrix_a, vector_b


def read_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content


def read():
    filename = "../NMLab1/lab2/input/matrix.txt"
    mtx_content = read_file(filename)
    return parse_input(mtx_content)
