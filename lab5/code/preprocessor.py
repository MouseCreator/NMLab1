import numpy as np
import val as v


def optimal_values(function, a, b, n):
    if a > b:
        raise ValueError("Illegal range!")
    vals_arr = []
    ab_sum = (a + b) / 2
    ab_diff = (b - a) / 2
    for i in range(n):
        x = ab_sum + ab_diff * np.cos((2 * i + 1) * np.pi / (2 * (n + 1)))
        f = v.at(function, x)
        vals_arr.append(v.Val([x, f]))
    return vals_arr


def even_values(function, a, b, n):
    if a > b:
        raise ValueError("Illegal range!")
    vals_arr = []
    if n == 1:
        x = (a+b)/2
        f = v.at(function, x)
        return [v.Val([x, f])]
    step = (b - a) / (n - 1)
    curr = a
    for i in range(n):
        x = curr
        curr += step
        f = v.at(function, x)
        vals_arr.append(v.Val([x, f]))
    return vals_arr
