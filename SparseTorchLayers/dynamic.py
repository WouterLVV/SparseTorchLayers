from abc import ABC

import torch
from SparseTorchLayers.base import SparseLayer

from SparseTorchLayers.F.generators.erdosrenyi import erdos_renyi
from SparseTorchLayers.F.selectors.magnitude import select_lowest_magnitude


class DynamicSparseLayer(SparseLayer):
    def __init__(self, in_size, out_size, **kwargs):
        super(DynamicSparseLayer, self).__init__(in_size, out_size, **kwargs)

        self.optimizer = None

    def update(self):
        self._coalesce()
        inds = self.mark_connections()
        self.remove_connections(inds)
        conns = self.generate_connections()
        self.add_connections(conns)

    def mark_connections(self):
        raise NotImplementedError

    def generate_connections(self):
        raise NotImplementedError

    @torch.no_grad()
    def remove_connections(self, inds):
        self.weight.data = self.weight.data[inds]
        self.idx.data = self.idx.data[:, inds]

        if self.optimizer is not None:
            for buffer in self.optimizer.state[self.weight]:
                self.optimizer.state[self.weight][buffer] = self.optimizer.state[self.weight][buffer][inds]

    @torch.no_grad()
    def add_connections(self, new_idx):
        # new_idx = self.edge_generator(num_to_regrow, self.in_size, self.out_size, existing=(self.idx, self.weight))
        num_new = new_idx.shape[1]
        new_idx.to(self.idx.device)
        self.idx.data = torch.cat((self.idx.data, new_idx), dim=1)
        new_weight = self.weight_generator((num_new,)).to(self.weight.device)
        self.weight.data = torch.cat((self.weight.data, new_weight))

        if self.optimizer is not None:
            zeros = torch.zeros((num_new,))
            for buffer in self.optimizer.state[self.weight]:
                self.optimizer.state[self.weight][buffer] = torch.cat((self.optimizer.state[self.weight][buffer], zeros))


class ConstantDensityDynamicSparseLayer(DynamicSparseLayer):
    def __init__(self, in_size, out_size, zeta, **kwargs):
        if "edge_generator" not in kwargs:
            kwargs["edge_generator"] = erdos_renyi
        super(ConstantDensityDynamicSparseLayer, self).__init__(in_size, out_size, **kwargs)
        self.zeta = zeta
        self.num_dynamic = int(self.zeta * self.nelement())

    def mark_connections(self):
        raise NotImplementedError

    def generate_connections(self):
        raise NotImplementedError


class SETLayer(ConstantDensityDynamicSparseLayer):
    def __init__(self, in_size, out_size, zeta=0.2, **kwargs):
        if "edge_generator" not in kwargs:
            kwargs["edge_generator"] = erdos_renyi
        super(SETLayer, self).__init__(in_size, out_size, zeta, **kwargs)

    def mark_connections(self):
        return select_lowest_magnitude(self.num_dynamic, self.weight)

    def generate_connections(self):
        return self.edge_generator((self.num_dynamic,))
