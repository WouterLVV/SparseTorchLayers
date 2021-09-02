import torch


def select_highest_magnitude(num_select, weight):
    mag = abs(weight)
    _, inds = torch.topk(mag, num_select, sorted=False)
    return inds


def select_lowest_magnitude(num_select, weight):
    mag = abs(weight)
    mag = 1./mag
    _, inds = torch.topk(mag, num_select, sorted=False)
    return inds

