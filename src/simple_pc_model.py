import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PCLayer(nn.Module):
    """Activity + energy at this point in the graph. Shape-agnostic."""

    def __init__(self):
        super().__init__()
        self.x = None
        self.energy_value = None

    def reset(self):
        self.x = None
        self.energy_value = None

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu

        if self.x is None or self.x.shape != mu.shape or self.x.device != mu.device:
            self.x = nn.Parameter(mu.detach().clone())

        self.energy_value = 0.5 * ((self.x - mu) ** 2).sum()
        return self.x


def pc_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, PCLayer)]


def activity_params(model: nn.Module):
    return [m.x for m in pc_layers(model) if m.x is not None]


def weight_params(model: nn.Module):
    act_ids = {id(p) for p in activity_params(model)}
    return [p for p in model.parameters() if id(p) not in act_ids]


def pc_energy(model: nn.Module) -> torch.Tensor:
    energies = [m.energy_value for m in pc_layers(model) if m.energy_value is not None]
    if not energies:
        return torch.tensor(0.0)
    return torch.stack([e.reshape(()) for e in energies]).sum()


def reset_pc(model: nn.Module):
    for m in pc_layers(model):
        m.reset()


# Any mix of Linear / Conv / activations. Put PCLayer where you want inferred x.
model = nn.Sequential(
    nn.Linear(1, 2),
    PCLayer(),
    nn.ReLU(),
    nn.Linear(2, 1),
)

# Conv example (same trainer):
# model = nn.Sequential(
#     nn.Conv2d(1, 8, 3, padding=1),
#     PCLayer(),
#     nn.ReLU(),
#     nn.Conv2d(8, 16, 3, padding=1),
#     PCLayer(),
#     nn.ReLU(),
#     nn.AdaptiveAvgPool2d(1),
#     nn.Flatten(),
#     nn.Linear(16, 1),
# )

inputs = [1.0, 2.0, 3.0, 4.0, 5.0]
targets = [2.0, 3.0, 4.0, 5.0, 6.0]
epochs = 10
N_relax = 100
step_x = 0.1
lr = 0.05

model.train()
opt_w = (
    None  # created after first forward (so Linear params exist; x exists after forward)
)

for epoch in range(epochs):
    for x0, y in zip(inputs, targets):
        x = torch.tensor([x0], dtype=torch.float32)
        target = torch.tensor([y], dtype=torch.float32)

        reset_pc(model)

        # Phase 1 — infer activities
        for _ in range(N_relax):
            out = model(x)
            E = pc_energy(model) + 0.5 * ((out - target) ** 2).sum()
            E.backward()

            with torch.no_grad():
                for a in activity_params(model):
                    if a.grad is not None:
                        a -= step_x * a.grad
                        a.grad.zero_()
            model.zero_grad()

        # Phase 2 — update weights / biases of Linear, Conv, etc.
        if opt_w is None:
            opt_w = torch.optim.SGD(weight_params(model), lr=lr)

        opt_w.zero_grad()
        out = model(x)
        E = pc_energy(model) + 0.5 * ((out - target) ** 2).sum()
        E.backward()
        opt_w.step()


@torch.no_grad()
def predict(x, model):
    was_training = model.training
    model.eval()  # PCLayer becomes identity
    y = model(torch.as_tensor(x, dtype=torch.float32).reshape(-1))
    model.train(was_training)
    return y


preds = [predict([v], model).item() for v in inputs]
print("predictions:", preds)

plt.figure(figsize=(8, 5))
plt.plot(inputs, targets, "o-", label="Targets")
plt.plot(inputs, preds, "x--", label="Predictions")
plt.legend()
plt.grid(True)
plt.show()
