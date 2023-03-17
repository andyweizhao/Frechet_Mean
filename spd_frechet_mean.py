
import numpy as np
import scipy


def frechet_mean(x):
    mu = x[0]
    loss_over_iterations = []
    epsilon = 1.
    while epsilon > 1e-12:
        assert len(loss_over_iterations) < 1000
        delta_mu = sum(riemannian_log(mu, xi) for xi in x) / len(x)
        new_mu = riemannian_exp(mu, delta_mu)
        epsilon = dist_squared(mu, new_mu) ** .5
        loss_over_iterations.append(sum(dist_squared(xi, mu) for xi in x))    
        mu = new_mu
    return mu, loss_over_iterations


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
    D = np.diag(np.random.rand(n) * max_eigenvalue)
    M = scipy.stats.ortho_group.rvs(dim=n)
    return M @ D @ M.transpose()


def example():
    n = 8  # dimension size
    t = 100  # number of points
    x = [random_spd_matrix(n) for _ in range(t)]
    mu, loss_over_iterations = frechet_mean(x)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    color = 'tab:blue'
    ax.plot(loss_over_iterations, linestyle='-', color=color)
    ax.set_xlabel('Iterations');
    ax.set_ylabel('$\sum_i d(x, x_i)^2$');
    ax.set_title(f'SPD_{n}, num_points = {t}')

