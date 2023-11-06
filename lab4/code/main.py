from newton import test_newton
from newton import test_newton_n_space
from newton import test_is_solvable
from iteration import test_relaxation

if __name__ == "__main__":
    eps = 1e-5
    print("Newton method 2x2")
    test_newton(eps)
    print("Newton method 10x10")
    test_newton_n_space(10, eps)
    print("Solvable test")
    test_is_solvable(eps)
    print("Relaxation method 2x2")
    test_relaxation(eps)