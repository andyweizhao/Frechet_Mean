import torch
import forward
import backward


class FrechetMean(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w):
        mean = forward.fr_mean_forward(x, w, atol=1e-6)
        ctx.save_for_backward(x, mean, w)
        return mean

    @staticmethod
    def backward(ctx, grad_output):
        x, mean, w = ctx.saved_tensors
        dx = backward.fr_mean_backward(x, mean, w, grad_output)
        return dx, None


def fr_mean(x, w=None):
    if w is None:
        w = torch.ones(x.size()[:-1]).to(x)
    return FrechetMean.apply(x, w)
