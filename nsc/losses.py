import torch
import torch.nn as nn
import numpy as np


def total_loss(model, x, t, e, weight_balance=1., eps=1e-10):
    # Go through network
    cumulative, intensity, alphas = model.forward(x, t, gradient=True)
    with torch.no_grad():
        intensity.clamp_(eps)

    # Likelihood error
    alphas = nn.LogSoftmax(dim=1)(alphas)
    cum = alphas - cumulative.sum(1)  # Sum over all risks

    # Sum over the different mixture and then across patient
    error = - weight_balance * torch.logsumexp(cum[e == 0], dim=1).sum()

    for r in range(model.risks):
        i = intensity[e == (r + 1)][:, r]
        error -= torch.logsumexp(cum[e == (r + 1)] + torch.log(i), dim=1).sum()

    return error / len(x)
