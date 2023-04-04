import torch
import geoopt.linalg.batch_linalg as lalg


def logmap(x, y):
    sqrt_x_inv, sqrt_x = lalg.sym_inv_sqrtm2(x)
    return sqrt_x @ lalg.sym_logm(sqrt_x_inv @ y @ sqrt_x_inv) @ sqrt_x


def expmap(x, y):
    sqrt_x_inv, sqrt_x = lalg.sym_inv_sqrtm2(x)
    return sqrt_x @ lalg.sym_expm(sqrt_x_inv @ y @ sqrt_x_inv) @ sqrt_x

def distance_squared(x, y):
    x_inv_sqrt = lalg.sym_inv_sqrtm1(x)
    s = x_inv_sqrt @ y @ x_inv_sqrt
    e = torch.linalg.eigh(s)[0]
    return (torch.log(e) ** 2).sum(dim=-1)


def expmap_approx(x, y):
    # A second order approximation of the exponential map above.
    return x + y + .5 * y @ torch.inverse(x) @ y


def logmap_for_forward(x, y):
    l_y = torch.linalg.cholesky(y)
    l_y_inv = torch.inverse(l_y)
    l, v = torch.linalg.eigh(l_y_inv @ x @ l_y_inv.transpose(-1, -2))
    return - l_y @ v @ torch.diag_embed(torch.log(l)) @ v.transpose(-1, -2) @ l_y_inv @ x, l.max(dim=-1)[0] / l.min(dim=-1)[0]


def to_coordinates(x):
    d = x.size(dim=-1)
    return x[..., torch.ones(d, d).triu() == 1]


def to_matrix(x):
    n = x.size(dim=-1)
    d = int(((1 + 8 * n) ** .5 - 1) / 2)
    assert d * (d + 1) == 2 * n
    res = torch.zeros(*(x.size()[:-1]), d, d, dtype=x.dtype)
    i, j = torch.triu_indices(d, d)
    res[..., i, j] = x
    res[..., j, i] = x
    return res


def riemannian_norm(x, v):
    x_inv = torch.inverse(x)
    return lalg.trace(x_inv @ v @ x_inv @ v) ** .5


def fr_mean_forward(x, w, max_iter=150, atol=1e-10, verbose=False):
    """
    The implementation is based on algorithm 3 in https://www.math.fsu.edu/~whuang2/pdf/KarcherMeanSPD_techrep.pdf .
    
    Args
    ----
    x (tensor) [..., points, dim]    (usually it will be either [points, dim] or [batch_size, points, dim])
    w (tensor) [..., points]

    Note: the points in SPD are given in coordinate form, that is by the vectorized upper triangle.

    Returns
    ---
    frechet mean (tensor) [..., dim]
    """
    x_mat = to_matrix(x)
    mu = x_mat[..., 0, :, :].clone()
    iters = 0
    mask = torch.ones(size=mu.shape[:-2]).bool()
    while iters < max_iter:
        if not mask.any():
            break
        g, c = logmap_for_forward(mu[mask].unsqueeze(-3), x_mat[mask])
        s = (g * w[mask][..., None, None]).sum(dim=-3) / w[mask].sum(dim=-1)[..., None, None]

        log_c = .5 * torch.log(c)
        delta = (w[mask] * torch.nan_to_num(log_c * torch.cosh(log_c) / torch.sinh(log_c))).sum(dim=-1) / w[mask].sum(dim=-1)
        alpha = 1. / delta
        mu[mask] = expmap_approx(mu[mask], alpha[..., None, None] * s)

        norm = riemannian_norm(mu[mask], s)
        new_mask = norm > atol
        mask[mask.clone()] = new_mask
        if verbose:
            print("step:", iters, " norm:", norm.max())
        iters += 1
    if iters == max_iter:
        print("Warning: has reached maximum iterations:", iters)
    else:
        print("Number of iterations:", iters)
    return to_coordinates(mu)


