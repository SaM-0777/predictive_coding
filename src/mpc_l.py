from typing import Callable

import torch
import torch.nn as nn


class PCLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._sample_x: bool = False
        self._x: torch.Tensor | None = None
        self._energy: torch.Tensor | None = None
        self.eval()

    def _sample_x_fn(self, x_hat: torch.Tensor) -> torch.Tensor:
        return x_hat.detach().clone()

    def _energy_fn(self, x_hat: torch.Tensor) -> torch.Tensor:
        assert self._x is not None, "self._x has not been initialized yet"
        energy = 0.5 * ((x_hat - self._x) ** 2).sum()
        self._energy = energy
        return energy

    def _clear_energy(self) -> None:
        self._energy = None

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x_hat

        if not self._sample_x:
            if self._x is None:
                self._sample_x = True

            elif self._x.device != x_hat.device:
                self._sample_x = True

            elif self._x.size() != x_hat.size():
                self._sample_x = True

        if self._sample_x:
            x = self._sample_x_fn(x_hat)
            self._x = nn.Parameter(data=x.data.to(x_hat.device), requires_grad=True)
            self._sample_x = False

        self._energy_fn(x_hat)
        assert self._x is not None, "self._x has not been initialized yet"
        return self._x
