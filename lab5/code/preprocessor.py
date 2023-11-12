import numpy as np
import val as v


def optimal_values(function, a, b, n):
    if a > b:
        raise ValueError("Illegal range!")
    vals_arr = []
    ab_sum = (a + b) / 2
    ab_diff = (b - a ) / 2
    for i in range(n):
        x = ab_sum + ab_diff * np.cos((2*i+1)*np.pi / (2*(n+1)))
        f = v.at(function, x)
        vals_arr.append([x, f])
    return vals_arr
