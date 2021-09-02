import torch


# idx0 must have unique entries
def subtract_edges(idx0, idx1):
    concat = torch.cat((idx0, idx1), dim=1)
    uniq, reverse_indices, counts = torch.unique(concat, dim=1, sorted=True, return_counts=True, return_inverse=True)
    uniq[:, counts > 1] = -1  # mark duplicates
    idx2 = uniq[:, reverse_indices[:idx0.shape[1]]]
    idx2 = idx2[:, idx2[0] != -1]
    return idx2
