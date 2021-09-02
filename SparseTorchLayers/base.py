import math

import torch
from torch import nn
from torch_sparse import coalesce, spmm, transpose

from SparseTorchLayers.F.generators.erdosrenyi import erdos_renyi
from SparseTorchLayers.F.selectors.magnitude import select_highest_magnitude


def _normal_generator(avg, std):
    def func(size):
        return torch.randn(size) * std + avg
    return func


class SparseLayer(torch.nn.Module):
    # @torch.no_grad()
    def __init__(self, in_size, out_size, weight_generator, edge_generator, initial_density=None, bias=True, graph_data=None, dtype=None):

        if (initial_density is None) == (graph_data is None):
            raise Exception("Must define either density or graph data")

        super(SparseLayer, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.weight = None

        if weight_generator is not None:
            if callable(weight_generator):
                self.weight_generator = weight_generator
            else:
                self.weight_generator = _normal_generator(weight_generator[0], weight_generator[1])
        else:
            self.weight_generator = _normal_generator(0, math.sqrt(2 / self.in_size))

        if edge_generator is not None:
            self.edge_generator = edge_generator
        else:
            self.edge_generator = erdos_renyi

        # with torch.no_grad():
        if graph_data is None:
            num_edges = min(int(self.in_size * self.out_size * initial_density), self.in_size * self.out_size)
            graph_data = self.edge_generator(num_edges, self.in_size, self.out_size)
        else:
            if isinstance(graph_data, SparseLayer):
                if bias and graph_data.bias is not None:
                    self.bias = graph_data.bias.clone().detach()
                graph_data = torch.stack((graph_data.idx.clone().detach(), graph_data.weight.clone().detach()))
            elif not isinstance(graph_data, torch.Tensor):
                torch.tensor(graph_data)

        if graph_data.shape[0] == 3:
            i = graph_data[:2]
            v = graph_data[2]
        else:
            i = graph_data
            v = self.weight_generator((i.shape[1],))

        self.idx = nn.Parameter(i, requires_grad=False)
        self.weight = nn.Parameter(v, requires_grad=True)
        self._coalesce()

        self.num_edges = len(self.weight)
        self.density = self.num_edges / (self.in_size * self.out_size)

        self.density = len(self.weight) / self.in_size * self.out_size

        if bias:
            if hasattr(self, "bias"):
                self.bias = nn.Parameter(self.bias, requires_grad=True)
            else:
                self.bias = nn.Parameter(self.weight_generator((1, self.out_size)), requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def _coalesce(self):
        # self.idx, self.weight = coalesce(self.idx, self.weight, self.in_size, self.out_size)
        new_idx, new_weight = coalesce(self.idx.data, self.weight.data, self.in_size, self.out_size)
        self.idx.data = new_idx
        self.weight.data = new_weight

    def nelement(self):
        if self.weight is None:
            return 0
        else:
            self._coalesce()
            return len(self.weight)

    def forward(self, x):
        idx_t, _ = transpose(self.idx, None, None, None, coalesced=False)
        z = spmm(idx_t, self.weight, self.out_size, self.in_size, x.t()).t()
        if self.bias is not None:
            z += self.bias
        return z





