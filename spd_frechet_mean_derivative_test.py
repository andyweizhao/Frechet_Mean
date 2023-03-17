import numpy as np
from spd_frechet_mean import *
from spd_frechet_mean_derivative import *
import time


def numerical_grad_d2y(x, y, epsilon=1e-4):
    d = x.shape[0]
    s = dist_squared(x, y)
    return np.array([(dist_squared(x + epsilon * e, y) - s) / epsilon for e in coordinate_matrices(d)])


def numerical_hess_d2y_matrix(x, y, epsilon=1e-4):
    d = x.shape[0]
    cm = coordinate_matrices(d)
    grad = grad_d2y(x, y)
    return np.reshape(np.concatenate([(np.array(grad_d2y(x + epsilon * f, y)) - grad) / epsilon for f in cm]),
                      (len(cm), len(cm)))


def numerical_mixed_second_der(x, y, epsilon=1e-4):
    d = x.shape[0]
    cm = coordinate_matrices(d)
    grad = grad_d2y(x, y)
    return np.reshape(np.concatenate([(np.array(grad_d2y(x, y + epsilon * f)) - grad) / epsilon for f in cm]), (len(cm), len(cm))).transpose()


def numerical_d_frechet_mean(x, epsilon=1e-5):
    d = x[0].shape[0]
    y0 = frechet_mean(x)[0]
    cm = coordinate_matrices(d)
    partials = []
    for i in range(len(x)):
        for e in cm:
            x[i] += epsilon * e  # not very good code
            y1 = frechet_mean(x)[0]
            x[i] -= epsilon * e
            der = (y1 - y0) / epsilon
            partials.append(der[np.triu_indices(d)])
    return np.array(partials).transpose()


def tests():
    d = 8
    x = random_spd_matrix(d, max_eigenvalue=1e3)
    y = random_spd_matrix(d, max_eigenvalue=1e3)
    print(np.linalg.norm(grad_d2y(x, y) - numerical_grad_d2y(x, y)))
    print(np.linalg.norm(hess_d2y_matrix(x, y) - numerical_hess_d2y_matrix(x, y)))
    print(np.linalg.norm(numerical_mixed_second_der(x, y) - mixed_second_derivatives_d2(x, y)))
    x = [random_spd_matrix(3, max_eigenvalue=1e3) for _ in range(3)]
    print(np.linalg.norm(numerical_d_frechet_mean(x) - d_frechet_mean(x)))


def measure_runtime(d, t):
    x = [random_spd_matrix(d, max_eigenvalue=1e3) for _ in range(t)]
    st = time.time()
    d_frechet_mean(x)
    print("d=", d, "  t=", t, "  time:", time.time() - st)


d = 8  # dimension size
t = 100  # number of points
tests()
measure_runtime(d, t)

""" # current speed: for d=8, t=1000 about 4 seconds (not including finding the frechet mean itself!) """
