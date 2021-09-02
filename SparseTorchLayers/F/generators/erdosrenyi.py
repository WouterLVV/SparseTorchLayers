import torch
from SparseTorchLayers.utils import subtract_edges


def erdos_renyi(num_edges, in_size, out_size, existing=None, extra=1.5):
    """Generate a random set of edges.

    :return A (2, num_edges) tensor with no duplicate edges.
    """

    num_existing = existing[0].shape[1] if existing is not None else 0
    num_max = in_size * out_size

    if num_edges + num_existing > num_max:
        raise Exception(f"Not enough unique elements possible. "
                        f"Requested: {num_edges + num_existing}, "
                        f"possible: {num_max}")

    elif num_edges + num_existing > num_max * 0.8:
        i = torch.arange(0, in_size).repeat_interleave(out_size)
        o = torch.arange(0, out_size).tile(in_size)
        edges = torch.stack((i, o))
    else:
        i = torch.randint(in_size, size=(int(num_edges * extra),))
        o = torch.randint(out_size, size=(int(num_edges * extra),))
        edges = torch.stack((i, o))
        edges = torch.unique(edges, dim=1, sorted=True)  # dim sorts the tensor anyway, sorted makes it explicit

    if num_existing > 0:
        edges = edges.to(existing[0].device)
        edges = subtract_edges(edges, existing[0])

    if edges.shape[1] < num_edges:
        return erdos_renyi(num_edges, in_size, out_size, existing, extra * 1.5)
    elif edges.shape[1] == num_edges:
        return edges
    else:
        return edges[:, torch.randperm(edges.shape[1])][:, :num_edges]
