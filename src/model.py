from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        activation_fn=torch.relu,
        activation_deriv=lambda a: (a > 0).float(),
    ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv

    def forward(self, x_above: torch.Tensor):
        a = x_above @ self.W.T
        x_hat = self.activation_fn(a)

        return x_hat, a


class PredictiveCodingNetwork(nn.Module):
    def __init__(self, dims, output_dim):
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1
        self.layers = nn.ModuleList(
            [PCNLayer(in_dim=dims[l + 1], out_dim=dims[l]) for l in range(self.L)]
        )
        self.readout = nn.Linear(dims[-1], output_dim, bias=False)

    def init_latents(self, batch_size, device):
        return [
            torch.randn(batch_size, d, device=device, requires_grad=False)
            for d in self.dims[1:]
        ]

    def compute_errors(self, inputs_latents):
        errors, gain_modulated_errors = [], []

        for l, layer in enumerate(self.layers):
            x_hat, a = layer(inputs_latents[l + 1])
            err = inputs_latents[l] - x_hat
            gm_err = err * layer.activation_deriv(a)
            errors.append(err)
            gain_modulated_errors.append(gm_err)

        return errors, gain_modulated_errors


def train_pcn(
    model,
    data_loader,
    num_epochs,
    eta_infer,
    eta_learn,
    T_infer,
    T_learn,
    device="mps",
):
    model.to(device).train()

    for epoch in range(num_epochs):
        for x_batch, y_batch in data_loader:
            B = x_batch.size(0)
            d_0 = model.dims[0]
            x_batch = x_batch.view(B, d_0).to(device)
            y_batch = (
                F.one_hot(y_batch, num_classes=model.readout.out_features)
                .float()
                .to(device)
            )
            inputs_latents = [x_batch] + model.init_latents(B, device)
            weigths = [layer.W for layer in model.layers] + [model.readout.weight]

            # Phase 1: INFERENCE for T steps
            with torch.no_grad(), torch.autocast(device):
                for t in range(1, T_infer + 1):
                    errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                    y_hat = model.readout(inputs_latents[-1])
                    eps_sup = y_hat - y_batch
                    eps_L = eps_sup @ weigths[-1]
                    errors_extended = errors + [eps_L]

                    for l in range(1, model.L + 1):
                        grad_Xl = (
                            errors_extended[l]
                            - gain_modulated_errors[l - 1] @ weigths[l - 1]
                        )
                        inputs_latents[l] -= eta_infer * grad_Xl

            # Phase 2: Learning
            with torch.no_grad():
                errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                y_hat = model.readout(inputs_latents[-1])
                eps_sup = y_hat - y_batch

                for l in range(model.L):
                    grad_Wl = -(gain_modulated_errors[l].T @ inputs_latents[l + 1]) / B
                    weigths[l] -= eta_learn * grad_Wl

                grad_Wout = eps_sup.T @ inputs_latents[-1] / B
                weigths[-1] -= eta_learn * grad_Wout


class PCLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        pc_backward: Callable[
            [nn.Module, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
    ) -> None:
        super().__init__()
        self.layer: nn.Module = layer
        self.pc_backward = pc_backward

        self.mu: torch.Tensor | None = None  # corresponds to mu j + 1
        self.out: torch.Tensor | None = None  # corresponds to out j + 1
        self.prediction_error: torch.Tensor | None = None  # corresponds to e j + 1
        self.prediction: torch.Tensor | None = None  # corresponds to j

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out: torch.Tensor = self.layer(x, *args, **kwargs)
        self.mu = out
        self.out = out.detach().clone()
        return out

    def backward(self, e: torch.Tensor) -> torch.Tensor:
        if self.mu is None:
            raise RuntimeError(
                f"PCLayer must perform a forward pass before PC inference."
            )

        return self.pc_backward(self.layer, self.mu, e)


class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            PCLayer(nn.Linear(in_dim, h_dim), pc_backward=linear_pc_backward),
            PCLayer(nn.Linear(h_dim, out_dim), pc_backward=linear_pc_backward),
        )
        # self.layer1 = nn.Linear(in_dim, h_dim)
        # self.layer2 = nn.Linear(h_dim, out_dim)


def linear_pc_backward(
    layer: nn.Module,
    mu: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    assert isinstance(layer, nn.Linear)
    return error @ layer.weight


def mse_deriv(out, label):
    return 2 * (out - label)


def relu(xs):
    return torch.clamp(xs, min=0)


def relu_deriv(xs):
    rel = relu(xs)
    rel[rel > 0] = 1
    return rel


def train(loss_fn_deriv, inference_lr: float = 0.1):
    T = 1

    in_dim = 2
    h_dim = 4
    out_dim = 2

    # x = torch.randn(in_dim)
    # y = torch.randn(out_dim)

    x = torch.tensor([0.8777, -0.4101])
    y = torch.tensor([0.6939, -0.8867])

    print(f"x {x}")
    print(f"y {y}")

    model = Model(in_dim, h_dim, out_dim)

    with torch.no_grad():
        x0 = x.detach().clone()
        out = x0

        for l in model.layers:
            out = l(out)
        output_layer = model.layers[-1]
        output_layer.mu = y.detach().clone()
        output_layer.prediction_error = -loss_fn_deriv(
            model.layers[-1].out,
            model.layers[-1].mu,
        )
        output_layer.prdicton = output_layer.prediction_error.detach().clone()

        for step in range(T):
            for j in reversed(range(1, len(model.layers))):
                layer = model.layers[j]
                lower_layer = model.layers[j - 1]

                layer.prediction_error = layer.mu - layer.out
                layer.prediction = layer.backward(layer.prediction_error)
                dx = lower_layer.prediction_error - layer.prediction_error
                lower_layer.mu -= inference_lr * (2 * dx)
            
            


if __name__ == "__main__":
    train(loss_fn_deriv=mse_deriv)
