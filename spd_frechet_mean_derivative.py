import numpy as np
import scipy
import spd_frechet_mean


def hess_d2y_matrix(x, y):
    """Calculates the Hessian matrix of the function x -> d(y, x)^2 (where y is fixed) in coordinates.
    (Note: this is not the Riemannian Hessian, but the one with respect to the flat connection.)
    """
    sqrt_y_inv = scipy.linalg.sqrtm(np.linalg.inv(y))

    def c(m): return sqrt_y_inv @ m @ sqrt_y_inv  # conjugation

    d = x.shape[0]
    cm = coordinate_matrices(d)
    l, v = np.linalg.eigh(c(x))

    def h(s, t):  # This function could be edited in order to improve conditioning.
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.isclose(s, t), (1 - np.log(s)) / s ** 2, (np.log(s) / s - np.log(t) / t) / (s - t))

    m = np.array([(v.transpose() @ c(e) @ v).reshape(d ** 2) for e in cm])
    i, j = np.indices((d, d))
    return 2 * (h(l[i], l[j]).reshape(d ** 2) * m) @ m.transpose()


def mixed_second_derivatives_d2(x, y):
    """Calculates the upper right block in the Hessian of d^2: SPD^2 -> R in coordinates, that is, the mixed second
       derivatives.
    """

    def sym(m): return (m + m.transpose()) / 2
    d = x.shape[0]
    cm = coordinate_matrices(d)

    # This block calculates the first term
    y_inv = np.linalg.inv(y)
    sqrt_y_inv = scipy.linalg.sqrtm(y_inv)
    l, v = np.linalg.eigh(sqrt_y_inv @ x @ sqrt_y_inv)
    a = sqrt_y_inv @ v @ np.diag(np.log(l) / l) @ v.transpose() @ sqrt_y_inv
    m1 = np.array([e.reshape(d * d) for e in cm])
    m2 = np.array([(y_inv @ e @ a).reshape(d * d) for e in cm])
    first_term = 2 * sym(m1 @ m2.transpose())

    # The second term can be expressed as a matrix product using the Hessian
    second_term = np.array([sym(f @ y_inv @ x)[np.triu_indices(d)] for f in cm]) @ hess_d2y_matrix(x, y)

    return - first_term - second_term


def d_frechet_mean(x):
    """This function calculates the Jacobian of the Frechet mean map. It implements the formula given in
    "Differentiating through the Frechet mean".
    """
    y = spd_frechet_mean.frechet_mean(x)[0]
    m1 = np.linalg.inv(sum(hess_d2y_matrix(y, xi) for xi in x))
    m2 = np.concatenate([mixed_second_derivatives_d2(xi, y).transpose() for xi in x], axis=1)
    return - m1 @ m2


def coordinate_matrices(d):
    """Returns a list of the d(d+1)/2 basis symmetric matrices. """

    def sym_coordinate_matrix(d, i, j):
        m = np.zeros((d, d))
        m[i, j] = 1
        m[j, i] = 1
        return m

    return [sym_coordinate_matrix(d, i, j) for i, j in zip(*np.triu_indices(d))]


def grad_d2y(x, y):
    """Calculates the gradient of the function x -> d(y, x)^2 in symmetric matrix coordinates (see below).
       Not used in the ultimate version of d_frechet_mean
    """
    d = x[0].shape[0]
    cm = coordinate_matrices(d)
    sqrt_y_inv = scipy.linalg.sqrtm(np.linalg.inv(y))

    def c(m): return sqrt_y_inv @ m @ sqrt_y_inv  # conjugation

    l, v = np.linalg.eigh(c(x))
    return np.array([2 * np.trace(v.transpose() @ c(e) @ v @ np.diag(np.log(l) / l)) for e in cm])