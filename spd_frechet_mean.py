import numpy as np
import scipy


def frechet_mean(x):
    mu = x[0]
    for _ in range(10):
        delta_mu = sum(riemannian_log(mu, xi) for xi in x) / len(x)
        new_mu = riemannian_exp(mu, delta_mu)
        print(((mu - new_mu) ** 2).sum(), sum(dist_squared(xi, mu) for xi in x))
        mu = new_mu
    return mu


def riemannian_log(P, Q):
    sqrt_P = scipy.linalg.sqrtm(P)
    sqrt_P_inv = scipy.linalg.sqrtm(np.linalg.inv(P))
    return sqrt_P @ scipy.linalg.logm(sqrt_P_inv @ Q @ sqrt_P_inv) @ sqrt_P


def riemannian_exp(P, Q):
    sqrt_P = scipy.linalg.sqrtm(P)
    sqrt_P_inv = scipy.linalg.sqrtm(np.linalg.inv(P))
    return sqrt_P @ scipy.linalg.expm(sqrt_P_inv @ Q @ sqrt_P_inv) @ sqrt_P


def dist_squared(P, Q):
    sqrt_P_inv = scipy.linalg.sqrtm(np.linalg.inv(P))
    return (np.log(np.linalg.eigh(sqrt_P_inv @ Q @ sqrt_P_inv)[0]) ** 2).sum()


def random_spd_matrix(n, max_eigenvalue=1):
    D = np.diag(np.random.rand(n) * max_eigenvalue + 1e-5)
    M = scipy.stats.ortho_group.rvs(dim=n)
    return M @ D @ M.transpose()


n = 20
t = 8
x = [random_spd_matrix(n) for _ in range(t)]
frechet_mean(x)
