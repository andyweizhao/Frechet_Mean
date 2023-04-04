import torch
from forward import to_coordinates, to_matrix
import geoopt.linalg.batch_linalg as lalg


def gradient_d2_y(x, y):
    """
    Args
    ___
    x, y tensor [..., dim] (in coordinate form)

    Returns
    ---
    tensor [..., dim]
    """
    x_mat = to_matrix(x)
    y_mat = to_matrix(y)
    if len(x.size()) < len(y.size()):
        x_mat = x_mat.unsqueeze(-3)

    y_inv_sqrt = lalg.sym_inv_sqrtm1(y_mat)
    x_conj = y_inv_sqrt @ x_mat @ y_inv_sqrt
    l, v = torch.linalg.eigh(x_conj)
    m = 4 * y_inv_sqrt @ v @ torch.diag_embed(torch.log(l) / l) @ v.transpose(-1, -2) @ y_inv_sqrt
    d = x_mat.size(-1)
    m[..., range(d), range(d)] *= .5   # special treatment for the diagonal
    return to_coordinates(m)


def hess_d2y(x, y):
    """
    Calculates the Hessian matrix of the function x -> d(x,y)^2.
    Args:
    ___
    x, y tensor [..., dim].

    Returns:
    [..., dim, dim]


    In the call from fr_mean_backward, it is used with broadcasting:
    x [..., dim]
    y [..., points, dim]
    """

    x_mat = to_matrix(x)
    y_mat = to_matrix(y)
    if len(x.size()) < len(y.size()):
        x_mat = x_mat.unsqueeze(-3)
    y_inv_sqrt = lalg.sym_inv_sqrtm1(y_mat)
    x_conj = y_inv_sqrt @ x_mat @ y_inv_sqrt
    l, v = torch.linalg.eigh(x_conj)
    d = x_mat.size(-1)
    unit_matrices = to_matrix(torch.eye(d * (d + 1) // 2).to(x))
    t = (y_inv_sqrt @ v).unsqueeze(-3)
    m = t.transpose(-1, -2) @ unit_matrices @ t

    l1, l2 = l.unsqueeze(-1), l.unsqueeze(-2)
    h = torch.where(abs(l1 - l2) > 1e-10, (torch.log(l2) / l2 - torch.log(l1) / l1) / (l2 - l1), (1. - torch.log(l1)) / l1 ** 2)

    res = 2 * ((h.unsqueeze(-3) * m).unsqueeze(-3) * m.unsqueeze(-4)).sum((-2, -1))
    return res


def second_term(x, y, grad_input):
    """Calculates the vector Jacobian product in the mixed derivatives.
    Args:
    ---
    x [..., points, dim]
    y [..., dim]
    grad_input [..., dim]

    Returns:
    [..., points, dim]
    """
    x_mat = to_matrix(x)
    y_mat = to_matrix(y)
    grad_input_mat = to_matrix(grad_input).unsqueeze(-3)

    y_inv_sqrt = lalg.sym_inv_sqrtm1(y_mat).unsqueeze(-3)
    x_conj = y_inv_sqrt @ x_mat @ y_inv_sqrt
    l, v = torch.linalg.eigh(x_conj)
    m = 2 * y_inv_sqrt @ v @ torch.diag_embed(torch.log(l) / l) @ v.transpose(-1, -2) @ y_inv_sqrt @ grad_input_mat @ torch.inverse(y_mat).unsqueeze(-3)
    d = x_mat.size(-1)
    m[..., range(d), range(d)] *= .5
    first_summand = m + m.transpose(-1, -2)

    t = y_inv_sqrt @ v
    l1, l2 = l.unsqueeze(-1), l.unsqueeze(-2)
    h = torch.where(abs(l1 - l2) > 1e-10, (torch.log(l2) / l2 - torch.log(l1) / l1) / (l2 - l1),
                    (1. - torch.log(l1)) / l1 ** 2)
    m = 2 * torch.inverse(y_mat).unsqueeze(-3) @ x_mat @ t @ (h * (t.transpose(-1, -2) @ grad_input_mat @ t)) @ t.transpose(-1, -2)
    m[..., range(d), range(d)] *= .5
    second_summand = m + m.transpose(-1, -2)

    return to_coordinates(- first_summand - second_summand)


def fr_mean_backward(x, y, w, input_grad):
    """
    Main function.

    Args
    ___
    x: tensor [..., points, dim]
    y: tensor [..., dim]

    Returns
    ---
    tensor [..., points, dim]
    """
    square_term = torch.inverse((w[..., None, None] * hess_d2y(y, x)).sum(dim=-3))
    out = - (square_term @ input_grad.unsqueeze(-1)).squeeze()
    out = second_term(x, y, out)
    return out

