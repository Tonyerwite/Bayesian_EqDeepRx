"""Gather CIR factors before their time expansion; preserve eager arithmetic."""
from dataclasses import dataclass
from types import MethodType

import torch


@dataclass
class _Factors:
    spatial: torch.Tensor
    doppler: torch.Tensor
    power: torch.Tensor

    def clusters(self, indices):
        def select(value):
            index = indices.reshape(*indices.shape, 1, 1, 1, 1)
            index = index.expand(*indices.shape, *value.shape[4:])
            return torch.gather(value, 3, index)
        # Same two complex multiplies, in the same order, as Sionna.
        return (select(self.spatial) * select(self.doppler)) * select(self.power)


def install_factor_gather(channel):
    """Patch only this cached Sionna instance, with a general eager fallback."""
    sampler = channel._cir_sampler
    original_nlos = sampler._step_11_nlos
    original_reduce = sampler._step_11_reduce_nlos

    def nlos(self, phi, topology, rays, t):
        if not self._subclustering or getattr(rays, 'blockage_loss_db', None) is not None:
            return original_nlos(phi, topology, rays, t)
        phase = self._step_11_phase_matrix(phi, rays)
        field = self._step_11_field_matrix(
            topology, rays.aoa, rays.aod, rays.zoa, rays.zod, phase)
        array = self._step_11_array_offsets(
            topology, rays.aoa, rays.aod, rays.zoa, rays.zod)
        doppler = self._step_11_doppler_matrix(
            topology, rays.aoa, rays.aod, rays.zoa, rays.zod, t)
        spatial = (field * array).unsqueeze(-1)
        power = torch.complex(torch.sqrt(rays.powers / spatial.shape[4]),
                              torch.zeros_like(rays.powers))
        for _ in range(spatial.dim() - power.dim()):
            power = power.unsqueeze(-1)
        return _Factors(spatial, doppler.unsqueeze(-2).unsqueeze(-2), power)

    def reduce(self, full, rays, c_ds):
        if not isinstance(full, _Factors):
            return original_reduce(full, rays, c_ds)
        powers, delays = rays.powers, rays.delays
        strongest_2 = getattr(rays, 'strongest_cluster_indices', None)
        if strongest_2 is None:
            strongest_clusters = torch.argsort(powers, dim=-1, descending=True)
            strongest_2 = strongest_clusters[..., :2]
        else:
            remaining = powers.scatter(-1, strongest_2,
                torch.full_like(strongest_2, -torch.inf, dtype=powers.dtype))
            rest = torch.argsort(remaining, dim=-1, descending=True)[..., :powers.shape[-1]-2]
            strongest_clusters = torch.cat([strongest_2, rest], dim=-1)
        sorted_delays = torch.gather(delays, 3, strongest_clusters)
        strong_delays, weak_delays = sorted_delays[..., :2], sorted_delays[..., 2:]
        offsets = self._sub_cl_delay_offsets.reshape([1]*(strong_delays.dim()-1)+[-1, 1])
        sub_delays = strong_delays.unsqueeze(-2) + offsets*c_ds.unsqueeze(-1).unsqueeze(-1)
        sub_delays = sub_delays.reshape(*sub_delays.shape[:-2], -1)
        strong = full.clusters(strongest_2)
        weak = full.clusters(strongest_clusters[..., 2:])
        sub1 = strong[:, :, :, :, self._sub_cl_1_ind, ...].sum(dim=4)
        sub2 = strong[:, :, :, :, self._sub_cl_2_ind, ...].sum(dim=4)
        sub3 = strong[:, :, :, :, self._sub_cl_3_ind, ...].sum(dim=4)
        weak = weak.sum(dim=4)
        result = torch.cat([sub1, sub2, sub3, weak], dim=3)
        delays = torch.cat([sub_delays, weak_delays], dim=3)
        order = torch.argsort(delays, dim=-1)
        delays = torch.gather(delays, 3, order)
        index = order.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        index = index.expand(-1, -1, -1, -1, *result.shape[4:])
        return torch.gather(result, 3, index), delays

    sampler._step_11_nlos = MethodType(nlos, sampler)
    sampler._step_11_reduce_nlos = MethodType(reduce, sampler)
