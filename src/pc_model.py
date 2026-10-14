from pathlib import Path
from typing import Callable, Iterable

from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

Activation = Callable[[torch.Tensor], torch.Tensor]


def mse_prediction_derivative(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Derivative of ``1/2 * ||target - prediction||^2`` w.r.t. prediction."""

    return prediction - target


class PCLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        activation_fn: Activation = lambda x: x,
        lr: float = 1e-4,
        gradient_clip: float | None = 50.0,
    ) -> None:
        super().__init__()

        self.layer = layer
        self.activation_fn = activation_fn
        self.lr = lr
        self.gradient_clip = gradient_clip
        self.input: torch.Tensor | None = None

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(self.layer(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward prediction of a generic differentiable module.

        Mathematical formulation:
            v_hat_j = f_j(v_i; theta_j)

        where:
            v_i       : input activity of the current module.
            f_j       : the module's differentiable prediction function.
            theta_j   : the module's parameters.
            v_hat_j   : predicted activity at the child vertex.

        The input activity is stored so that the local prediction function
        can be reconstructed when calculating its derivative during backward().

        Returns:
            v_hat_j: the module's predicted output.
        """
        self.input = x.detach().clone()
        return self._predict(x)

    def backward(self, prediction_error: torch.Tensor) -> torch.Tensor:
        """
        Calculate the derivative-weighted error contribution from a child
        prediction to its parent activity, using PyTorch autograd.

        Mathematical formulation:
            v_hat_j = f_j(v_i; theta_j)
            J_ji = d(v_hat_j) / d(v_i)
            dx = J_ji^T @ e

        For predictive coding, when e = epsilon_j, where epsilon_j is the
        child's prediction error, this calculates:

            dx = (d(v_hat_j) / d(v_i))^T @ epsilon_j

        This is the derivative-weighted child-error contribution appearing
        in Equation 2 of Millidge et al. (2020).

        The complete predictive-coding activity dynamics for parent vertex i,
        using the equation's notation, are:

            dv_i/dt = epsilon_i - sum_{j in C(v_i)} J_ji^T @ epsilon_j

        where:
            epsilon_i : local prediction error at the parent vertex.
            C(v_i)    : set of child vertices of the parent.
            J_ji      : Jacobian of child j's prediction with respect to
                        parent i's activity.

        This method calculates one child contribution. The caller must sum
        contributions from all children and combine them with the parent's
        local prediction error to implement the full activity equation.

        PyTorch autograd computes the vector-Jacobian product without
        explicitly constructing the Jacobian matrix.

        Args:
            prediction_error: Incoming error signal. For the predictive-coding interpretation,
                this must be the corresponding child's prediction error.

        Returns:
            dx: Derivative-weighted error contribution with respect to the
                module's input activity.
        """
        if self.input is None:
            raise RuntimeError("Call forward() before backward().")

        with torch.enable_grad():
            x = self.input.detach().requires_grad_(True)
            prediction = self._predict(x)
            (dx,) = torch.autograd.grad(
                outputs=prediction,
                inputs=x,
                grad_outputs=prediction_error.detach(),
                create_graph=False,
                retain_graph=False,
            )

        return dx.detach()

    def update_weights(
        self, prediction_error: torch.Tensor
    ) -> tuple[torch.Tensor, ...] | None:
        """
        Update the prediction-function parameters according to Equation 3
        of Millidge et al. (2020), Predictive Coding Approximates Backprop
        Along Arbitrary Computation Graphs.

        Paper's equations:

            v_hat_i = f_i(x; theta_i)
            epsilon_i = v_i - v_hat_i
            d(theta_i)/dt = -dF/d(theta_i)
                          = epsilon_i^T * d(v_hat_i)/d(theta_i)

        Discrete parameter update:

            theta_i <- theta_i + learning_rate * [d(v_hat_i)/d(theta_i)]^T * epsilon_i

        This method calculates the local parameter-update direction using
        PyTorch autograd and applies it after activity inference converges.

        Args:
            prediction_error:
                The local prediction error epsilon_i = v_i - v_hat_i
                associated with this prediction function's output.
        """
        # The prediction function requires its feedforward input x:
        # v_hat_i = f_i(x; theta_i).
        # The cached input is used to evaluate the local parameter derivative.
        if self.input is None:
            raise RuntimeError("Call forward() before update_weights().")

        # theta_i denotes the trainable parameters of the prediction function.
        # Equation 3 requires derivatives with respect to these parameters:
        # d(theta_i)/dt = -dF/d(theta_i).
        parameters = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

        if not parameters:
            return None

        # Treat the prediction-function input as fixed while differentiating
        # with respect to theta_i:
        # v_hat_i = f_i(x; theta_i).
        x = self.input.detach()

        with torch.enable_grad():
            # v_hat_i = activation_fn(layer(x; theta_i)).
            prediction = self._predict(x)

            # Calculate the local parameter-update direction:
            #
            # g_theta_i = [d(v_hat_i)/d(theta_i)]^T * epsilon_i.
            #
            # This is the derivative term in Equation 3.
            # PyTorch calculates the vector-Jacobian product directly.
            # prediction_error is held fixed during this derivative.
            dW = torch.autograd.grad(
                outputs=prediction,
                inputs=parameters,
                grad_outputs=prediction_error.detach(),
                allow_unused=True,
                create_graph=False,
                retain_graph=False,
            )

        # Apply the positive parameter-update direction specified by Equation 3:
        #
        # d(theta_i)/dt = epsilon_i^T * d(v_hat_i)/d(theta_i).
        #
        # torch.no_grad() ensures that the in-place parameter update does not
        # become part of another autograd computation graph.
        with torch.no_grad():
            for parameter, update in zip(parameters, dW):
                if update is None:
                    continue
                # A parameter independent of this prediction has no local
                # derivative contribution: d(v_hat_i)/d(parameter) = 0.
                update = 2.0 * update
                if self.gradient_clip is not None:
                    update = update.clamp(
                        min=-self.gradient_clip, max=self.gradient_clip
                    )
                    # Forward Euler parameter update:
                    #
                    # theta_i^(t+1) = theta_i^t + eta_theta * g_theta_i.
                parameter.add_(update, alpha=self.lr)

        return dW


class Model(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        T: int = 5,
        inference_lr: float = 0.1,
        parameter_lr: float = 1e-4,
        gradient_clip: float | None = 50.0,
        loss_fn_deriv: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = lambda prediction, activation: (prediction - activation),
        image_size: tuple[int, int] = (28, 28),
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.T = T
        self.inference_lr = inference_lr
        self.loss_fn_deriv = loss_fn_deriv
        self.image_size = image_size

        flattened_dim = 1024
        second_hidden_dim = 150

        self.layers = nn.Sequential(
            PCLayer(
                nn.Conv2d(in_dim, 6, kernel_size=5),
                torch.relu,
                lr=parameter_lr,
                gradient_clip=gradient_clip,
            ),
            PCLayer(
                nn.MaxPool2d(kernel_size=2),
                lambda x: x,
                lr=parameter_lr,
                gradient_clip=gradient_clip,
            ),
            PCLayer(
                nn.Conv2d(6, 16, kernel_size=5),
                torch.relu,
                lr=parameter_lr,
                gradient_clip=gradient_clip,
            ),
            PCLayer(
                nn.Flatten(),
                lambda x: x,
                lr=parameter_lr,
                gradient_clip=gradient_clip,
            ),
            PCLayer(
                nn.Linear(flattened_dim, hidden_dim),
                torch.relu,
                lr=parameter_lr,
                gradient_clip=gradient_clip,
            ),
            PCLayer(
                nn.Linear(hidden_dim, second_hidden_dim),
                torch.relu,
                lr=parameter_lr,
                gradient_clip=gradient_clip,
            ),
            PCLayer(
                nn.Linear(second_hidden_dim, out_dim),
                lambda x: x,
                lr=parameter_lr,
                gradient_clip=gradient_clip,
            ),
        )

        self.L = len(self.layers)
        self.predictions: list[torch.Tensor] = [
            torch.empty(0) for _ in range(self.L + 1)
        ]
        self.activations: list[torch.Tensor] = [
            torch.empty(0) for _ in range(self.L + 1)
        ]
        self.prediction_errors: list[torch.Tensor] = [
            torch.empty(0) for _ in range(self.L + 1)
        ]
        self.inference_error_signals: list[torch.Tensor] = [
            torch.empty(0) for _ in range(self.L + 1)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def initialize(self, x: torch.Tensor, y: torch.Tensor) -> None:
        with torch.no_grad():
            self.predictions[0] = x.detach().clone()
            self.activations[0] = x.detach().clone()

            for i, l in enumerate(self.layers):
                prediction = l(self.predictions[i])
                self.predictions[i + 1] = prediction.detach().clone()
                if i < self.L - 1:
                    self.activations[i + 1] = prediction.detach().clone()

            self.activations[self.L] = y.detach().clone()
            self.prediction_errors[self.L] = -self.loss_fn_deriv(
                self.predictions[self.L], self.activations[self.L]
            )
            self.inference_error_signals[self.L] = (
                self.prediction_errors[self.L].detach().clone()
            )
            self.prediction_errors[0] = torch.zeros_like(self.activations[0])
            self.inference_error_signals[0] = torch.zeros_like(self.activations[0])

    def inference(self):
        if self.predictions[0].numel() == 0:
            raise RuntimeError("Call initialize() before inference().")
        
        for _ in range(self.T):
            for j in reversed(range(self.L)):
                if j == 0:
                    continue
                
                self.prediction_errors[j] = (
                    self.activations[j] - self.predictions[j]
                )
                child_contribution = self.layers[j].backward(
                    self.prediction_errors[j + 1].detach()
                ) # type: ignore
                self.inference_error_signals[j] = child_contribution
                dx_l = self.prediction_errors[j] - child_contribution
                self.activations[j].add_(dx_l, alpha=-2.0 * self.inference_lr)

        # Refresh hidden errors after the final activity updates.
        # These errors will be used by Equation 3.
        for j in range(1, self.L):
            self.prediction_errors[j] = self.activations[j] - self.predictions[j]

    def update_weights(self):
        if self.predictions[0].numel() == 0:
            raise RuntimeError("Call initialize() before update_weights().")

        for i, l in enumerate(self.layers):
            dW = l.update_weights(self.prediction_errors[i + 1])  # type: ignore

    @staticmethod
    def _batch_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the paper-compatible per-batch squared prediction loss."""

        return 0.5 * (prediction - target).square().sum(dim=1).mean()

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        max_batches: int | None = None,
    ) -> tuple[float, float]:
        """Evaluate MSE and classification accuracy with a feedforward pass."""

        was_training = self.training
        super().eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch_index, (images, labels) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            target = F.one_hot(labels, num_classes=self.out_dim).to(images.dtype)
            prediction = self.forward(images)
            batch_size = labels.shape[0]
            total_loss += float(self._batch_loss(prediction, target)) * batch_size
            total_correct += int((prediction.argmax(dim=1) == labels).sum())
            total_examples += batch_size

        if was_training:
            super().train(True)
        if total_examples == 0:
            return 0.0, 0.0
        return total_loss / total_examples, total_correct / total_examples

    def pc_train(
        self,
        mode: bool = True,
        *,
        train_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        epochs: int = 1,
        device: torch.device | None = None,
        log_interval: int = 100,
        max_train_batches: int | None = None,
        max_test_batches: int | None = None,
    ):
        """Set module mode or run the complete predictive-coding loop."""

        if train_loader is None:
            return super().train(mode)

        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if device is None:
            device = next(self.parameters()).device

        super().train(True)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_correct = 0
            total_examples = 0

            for batch_index, (images, labels) in enumerate(train_loader):
                if max_train_batches is not None and batch_index >= max_train_batches:
                    break

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                target = F.one_hot(labels, num_classes=self.out_dim).to(images.device)

                self.initialize(images, target)
                self.inference()
                prediction = self.predictions[self.L]
                batch_loss = self._batch_loss(prediction, target)
                self.update_weights()

                batch_size = labels.shape[0]
                total_loss += float(batch_loss) * batch_size
                total_correct += int((prediction.argmax(dim=1) == labels).sum())
                total_examples += batch_size

                if log_interval > 0 and (batch_index + 1) % log_interval == 0:
                    print(
                        f"epoch {epoch} batch {batch_index + 1}: "
                        f"loss={float(batch_loss):.4f}"
                    )

            if total_examples == 0:
                raise RuntimeError("The training loader produced no batches.")

            train_loss = total_loss / total_examples
            train_accuracy = total_correct / total_examples
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)

            if test_loader is not None:
                test_loss, test_accuracy = self.evaluate(
                    test_loader, device, max_batches=max_test_batches
                )
            else:
                test_loss, test_accuracy = 0.0, 0.0
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_accuracy)

            print(
                f"epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f} "
                f"train_accuracy={train_accuracy:.4%}"
                + (
                    f" test_loss={test_loss:.4f} " f"test_accuracy={test_accuracy:.4%}"
                    if test_loader is not None
                    else ""
                )
            )

        return history


def train(
    data_root: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
    T: int = 20,
    inference_lr: float = 0.1,
    parameter_lr: float = 5e-4,
    hidden_dim: int = 200,
    max_train_batches: int | None = None,
    max_test_batches: int | None = None,
):
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / "data"
    data_root = Path(data_root)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=False,
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, transform=transform, download=False
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = Model(
        in_dim=1,
        hidden_dim=hidden_dim,
        out_dim=10,
        T=T,
        inference_lr=inference_lr,
        parameter_lr=parameter_lr,
        loss_fn_deriv=mse_prediction_derivative,
    ).to(device)

    history = model.pc_train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        device=device,
        max_train_batches=max_train_batches,
        max_test_batches=max_test_batches,
    )

    return model, history


if __name__ == "__main__":
    train()
