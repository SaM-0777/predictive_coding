from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import tqdm

from .mpc_l import PCLayer


@dataclass
class PCTrainingArguments:
    T: int = 20
    num_train_epochs: int = 10
    energy_coefficient: float = 1.0
    update_p_at: str = "last"  # "last" | "all"
    optimizer_x_cls: Callable = optim.SGD
    optimizer_x_kwargs: dict[str, Any] = field(default_factory=lambda: {"lr": 0.1})
    optimizer_p_cls: Callable = optim.Adam
    optimizer_p_kwargs: dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3})
    logging_steps: int = 10
    eval_strategy: str = "epoch"  # "no" | "epoch"
    device: str | torch.device | None = None
    max_grad_norm_x: float | None = None
    max_grad_norm_p: float | None = None


class PCTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: PCTrainingArguments | None = None,
        train_loader: DataLoader | None = None,
        eval_loader: DataLoader | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        compute_metrics: (
            Callable[[torch.Tensor, torch.Tensor], dict[str, float]] | None
        ) = None,
    ) -> None:
        self.model = model
        self.args = args or PCTrainingArguments()
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.loss_fn = loss_fn or self._default_loss_fn
        self.compute_metrics = compute_metrics

        if self.args.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.args.device)

        self.model.to(self.device)
        self.optimizer_x: optim.Optimizer | None = None
        self.optimizer_p: optim.Optimizer | None = None
        self.state = {
            "epoch": 0,
            "global_step": 0,
            "log_history": [],
        }

    @staticmethod
    def _default_loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * (output - target).pow(2).sum()

    def pc_layers(self) -> list[PCLayer]:
        return [m for m in self.model.modules() if isinstance(m, PCLayer)]

    def _assert_ready(self) -> None:
        if not self.pc_layers():
            raise ValueError("Model has no PCLayer modules.")
        if not self.model.training:
            raise RuntimeError("Call model.train() before trainer.train().")
        if any(not layer.training for layer in self.pc_layers()):
            raise RuntimeError("Every PCLayer must be in train mode.")

    def _activity_params(self) -> list[nn.Parameter]:
        xs = []
        for layer in self.pc_layers():
            if layer.x is None:
                raise RuntimeError(
                    "PCLayer.x is None. Forward once after set_sample_x()."
                )
            xs.append(layer.x)
        return xs

    def _weight_params(self) -> list[nn.Parameter]:
        act_ids = {id(p) for p in self._activity_params()}
        return [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in act_ids
        ]

    def _set_sample_x(self, sample_x: bool = True) -> None:
        for layer in self.pc_layers():
            layer.set_sample_x(sample_x)

    def _pc_energy(self) -> torch.Tensor:
        energies = [
            layer.energy for layer in self.pc_layers() if layer.energy is not None
        ]
        if not energies:
            return torch.zeros((), device=self.device)
        return torch.stack([e.reshape(()) for e in energies]).sum()

    def compute_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        supervised = self.loss_fn(output, target)
        energy = self._pc_energy()
        total = energy * self.args.energy_coefficient + supervised
        return {"loss": total, "energy": energy, "supervised": supervised}

    def _rebuild_optimizer_x(self) -> None:
        self.optimizer_x = self.args.optimizer_x_cls(
            self._activity_params(), **self.args.optimizer_x_kwargs
        )

    def _ensure_optimizer_p(self) -> None:
        if self.optimizer_p is None:
            params = self._weight_params()
            if not params:
                raise RuntimeError("No weight parameters found.")
            self.optimizer_p = self.args.optimizer_p_cls(
                params, **self.args.optimizer_p_kwargs
            )

    def _prepare_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            data, target = batch
        elif isinstance(batch, dict):
            data = batch.get("input_ids", batch.get("inputs", batch.get("data")))
            target = batch.get("labels", batch.get("target", batch.get("y")))
            if data is None or target is None:
                raise ValueError("Batch dict must contain inputs and labels.")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        return data.to(self.device), target.to(self.device)

    def training_step(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        """One batch: resample x, infer T steps, update weights."""
        self._set_sample_x(True)
        output = self.model(data)  # creates / resizes each PCLayer.x
        self._rebuild_optimizer_x()
        self._ensure_optimizer_p()

        assert self.optimizer_x is not None, ""
        assert self.optimizer_p is not None, ""

        last_logs: dict[str, float] = {}
        for t in range(self.args.T):
            output = self.model(data)
            losses = self.compute_loss(output, target)

            self.optimizer_x.zero_grad(set_to_none=True)
            if self.args.update_p_at == "all" or t == self.args.T - 1:
                self.optimizer_p.zero_grad(set_to_none=True)

            print(f"loss {losses["loss"]}")
            losses["loss"].backward()

            if self.args.max_grad_norm_x:
                torch.nn.utils.clip_grad_norm_(
                    self._activity_params(), self.args.max_grad_norm_x
                )
            self.optimizer_x.step()

            update_p = self.args.update_p_at == "all" or t == self.args.T - 1
            if update_p:
                if self.args.max_grad_norm_p:
                    nn.utils.clip_grad_norm_(
                        self._weight_params(), self.args.max_grad_norm_p
                    )
                self.optimizer_p.step()

            last_logs = {k: float(v.detach()) for k, v in losses.items()}
            last_logs["t"] = t

        return last_logs

    def train(self, train_loader: DataLoader | None = None) -> dict[str, Any]:
        loader = train_loader or self.train_loader
        if loader is None:
            raise ValueError("Pass a train_loader to train() or the constructor.")

        self.model.train()
        for layer in self.pc_layers():
            layer.train()
        self._assert_ready()

        args = self.args
        for epoch in range(args.num_train_epochs):
            self.state["epoch"] = epoch
            pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
            running = {"loss": 0.0, "energy": 0.0, "supervised": 0.0}
            n_batches = 0

            for batch in pbar:
                data, target = self._prepare_batch(batch)
                logs = self.training_step(data, target)

                self.state["global_step"] += 1
                n_batches += 1
                for key in ("loss", "energy", "supervised"):
                    running[key] += logs[key]

                if self.state["global_step"] % args.logging_steps == 0:
                    pbar.set_postfix(
                        loss=f"{logs['loss']:.4f}",
                        energy=f"{logs['energy']:.4f}",
                        sup=f"{logs['supervised']:.4f}",
                    )
                    self.state["log_history"].append(
                        {"epoch": epoch, "step": self.state["global_step"], **logs}
                    )

            epoch_logs = {k: v / max(n_batches, 1) for k, v in running.items()}
            epoch_logs["epoch"] = epoch
            if args.eval_strategy == "epoch" and self.eval_loader is not None:
                epoch_logs.update({f"eval_{k}": v for k, v in self.evaluate().items()})
            self.state["log_history"].append(epoch_logs)

        return self.state

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader | None = None) -> dict[str, float]:
        loader = eval_loader or self.eval_loader
        if loader is None:
            raise ValueError("No eval_loader provided.")

        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        n = 0
        metric_acc: dict[str, float] = {}
        all_out, all_y = [], []

        for batch in loader:
            data, target = self._prepare_batch(batch)
            output = self.model(data)
            loss = self.loss_fn(output, target)
            bs = data.shape[0] if data.ndim > 0 else 1
            total_loss += float(loss) * bs
            n += bs
            if self.compute_metrics is not None:
                all_out.append(output.cpu())
                all_y.append(target.cpu())

        logs = {"loss": total_loss / max(n, 1)}
        if self.compute_metrics is not None and all_out:
            metrics = self.compute_metrics(torch.cat(all_out), torch.cat(all_y))
            logs.update(metrics)
            metric_acc.update(metrics)

        self.model.train(was_training)
        for layer in self.pc_layers():
            layer.train(was_training)
        return logs

    @torch.no_grad()
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        out = self.model(data.to(self.device))
        self.model.train(was_training)
        for layer in self.pc_layers():
            layer.train(was_training)
        return out
