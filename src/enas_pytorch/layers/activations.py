import torch
import torch.nn.functional as F


def weighted_activation(weighting):
    def helper(x):
        return weighting[0] * F.relu(x) + weighting[1] * torch.tanh(x) + weighting[2] * x + weighting[3] * \
               torch.sigmoid(x)

    return helper