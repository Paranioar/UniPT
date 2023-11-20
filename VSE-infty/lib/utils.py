import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def count_params(model_parameters):
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def safe_random_list(_num, _rate):
    while True:
        rand_list = np.random.rand(_num)
        choice = np.where(rand_list > _rate)[0]
        if len(choice) >= 1:
            break
    return choice


def random_drop_feature(feature, rate):
    
    feature_after_drop = torch.zeros_like(feature).to(feature.device)
    lengths_after_drop = []
    for k in range(feature.size(0)):
        choice = safe_random_list(feature.size(1), rate)
        lengths_after_drop.append(len(choice))
        feature_after_drop[k, :len(choice)] = feature[k][choice]
    lengths_after_drop = torch.Tensor(lengths_after_drop).to(feature.device)

    return feature_after_drop, lengths_after_drop