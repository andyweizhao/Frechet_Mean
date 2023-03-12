"""
Simple implementation for the hyperbolic mean in the Poincaré disk with K = -1, R = 1

(Algorithm 1 from "Differentiating through the Fréchet Mean")
"""

import numpy as np


def example():
    """ Returns t random points in the Poincaré disk, along with their Euclidean and Frechet means
    the weights w can be set to other values that sum to 1
    """
    t = 4
    x = [random_point() for _ in range(t)]
    w = [1 / t] * t
    return x, eucl_mean(x, w), fr_mean(x, w)


def eucl_mean(x, w):
    return sum(wi * xi for wi, xi in zip(w, x))


def fr_mean(x, w):
    T = 15   # number of iterations
    y = x[0]
    for _ in range(T):
        a, b, c = (0, np.zeros(2), 0)
        for xl, wl in zip(x, w):
            alpha_l = wl * g(
                np.linalg.norm(xl - y, ord=2) / ((1 - np.linalg.norm(xl, ord=2)) * (1 - np.linalg.norm(y, ord=2)))) / (
                              1 - np.linalg.norm(xl, ord=2))
            a, b, c = (a + alpha_l, b + alpha_l * xl, c + alpha_l * np.linalg.norm(xl, ord=2))
        y = b * (a + c - ((a + c) ** 2 - 4 * np.linalg.norm(b, ord=2)) ** .5) / (2 * np.linalg.norm(b, ord=2))
    return y


def g(y):
    """g can be extended continuously to 0, but this has to be done explicitly"""
    if np.allclose(y, 0):
        return 4
    return 2 * np.arccosh(1 + 2 * y) / np.sqrt(y ** 2 + y)


def random_point():
    """Returns a point selected uniformly from a section of the unit disk"""
    p = np.zeros(2)
    r = np.random.uniform(.4, 1)             # possible radius
    phi = np.random.uniform(0, .7 * np.pi)   # possible angle
    p[0], p[1] = (r * np.cos(phi), r * np.sin(phi))
    return p
