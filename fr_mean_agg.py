import torch
import fr_mean
import torch.nn.functional as F
import forward


def frechet_agg(x, edge_index, edge_weight):
    """
    Compute Frechet aggregation.
    Inputs and outputs in coordinates.

    Args
    ---
    x tensor [points, dim]
    edge_index, edge_weight: sparse representation of the adjacency matrix, maximum index = points

    """
    n, d = x.size()

    b = max([(edge_index[0] == i).sum() for i in range(n)])

    batched_tensor = []
    weight_tensor = []

    d_ = forward.to_matrix(x[0]).shape[0]
    pad_matrix = torch.eye(d_).reshape(1, d_, d_)

    for i in range(n):
        si = edge_index[1, edge_index[0] == i]
        weight_tensor.append(F.pad(edge_weight[si], (0, b - len(si))))

        # pad with basepoints (identity matrix)
        pad_identity = forward.to_coordinates(pad_matrix.repeat(b - len(si), 1, 1))

        batched_tensor.append(torch.cat((x[si], pad_identity), dim=-2))
    batched_tensor = torch.stack(batched_tensor)
    weight_tensor = torch.stack(weight_tensor)

    return fr_mean.fr_mean(batched_tensor, w=weight_tensor)


def demo():
    # Very simple graph with random node features to show aggregation.

    number_of_points = 4
    dim = 4

    z = torch.matrix_exp(
        forward.to_matrix(torch.normal(mean=0., std=1., size=(number_of_points, dim * (dim + 1) // 2))))
    z = forward.to_coordinates(z).clone().detach().requires_grad_(True)

    print(torch.linalg.eigh(forward.to_matrix(z))[0])  # visualisation of eigenvalues

    edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 3],
                               [0, 1, 2, 3, 1, 3, 2, 3, 3]])
    w = torch.ones(edge_index.size(1))

    out = frechet_agg(z, edge_index, w)


#demo()
