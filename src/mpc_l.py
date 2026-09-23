from typing import Callable

import torch
import torch.nn as nn


class PCLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._sample_x: bool = False
        self._x: torch.Tensor | None = None
        self._energy: torch.Tensor | None = None
        self._energy_per_datapoint: torch.Tensor | None = None
        self.eval()

    @property
    def x(self) -> torch.Tensor | None:
        return self._x

    @property
    def energy(self) -> torch.Tensor | None:
        return self._energy

    @property
    def energy_per_datapoint(self) -> torch.Tensor | None:
        return self._energy_per_datapoint

    def set_sample_x(self, flag: bool = True) -> None:
        self._sample_x = bool(flag)

    def clear_energy(self) -> None:
        self._energy = None

    def clear_energy_per_datapoint(self) -> None:
        self._energy_per_datapoint = None

    def reset(self) -> None:
        self._x = None
        self._sample_x = False
        self.clear_energy()
        self.clear_energy_per_datapoint()

    def _sample_x_fn(self, x_hat: torch.Tensor) -> torch.Tensor:
        return x_hat.detach().clone()

    def _compute_energy(
        self, x_hat: torch.Tensor, energy_per_datapoint: bool = True
    ) -> torch.Tensor:
        assert self._x is not None, "self._x has not been initialized yet"
        energy = 0.5 * ((x_hat - self._x) ** 2)
        self._energy = (
            energy.sum()
        )  # total energy including all the batch data point for this layer, scalar
        if energy_per_datapoint:
            self._energy_per_datapoint = energy.sum(
                dim=list(range(energy.dim()))[1:],
                keepdim=False,
            ).unsqueeze(
                1
            )  # [B, 1]
            return self._energy_per_datapoint
        return self._energy

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x_hat

        need_sample = self._sample_x or self._x is None
        if self._x is not None and (
            self._x.device != x_hat.device or self._x.shape != x_hat.shape
        ):
            need_sample = True

        if need_sample:
            sampled = self._sample_x_fn(x_hat)
            self._x = nn.Parameter(data=sampled.to(x_hat.device), requires_grad=True)
            self._sample_x = False

        self._compute_energy(x_hat)
        assert self._x is not None, ""
        return self._x
