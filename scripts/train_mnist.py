import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from src.mpc_l import PCLayer
from src.mpc_trainer import PCTrainer, PCTrainingArguments

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# load data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]
)
train_dataset = datasets.MNIST("./data", download=True, train=True, transform=transform)
test_dataset = datasets.MNIST("./data", download=True, train=False, transform=transform)

batch_size = 500
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

print(f"# train images: {len(train_dataset)} and # test images: {len(test_dataset)}")


input_size = 28 * 28
hidden_size = 256
output_size = 10

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    PCLayer(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    PCLayer(),
    nn.Linear(hidden_size, output_size),
)
model.train()
model.to(device)

T = 20
epochs = 10


def loss_fn(yhat, y):
    y_oh = torch.nn.functional.one_hot(y.long(), num_classes=yhat.size(-1)).float()
    return 0.5 * (yhat - y_oh).pow(2).sum()


args = PCTrainingArguments(
    T=T,
    num_train_epochs=epochs,
    optimizer_x_kwargs={"lr": 0.1},
    optimizer_p_kwargs={"lr": 1e-3},
    update_p_at="last",
)

trainer = PCTrainer(
    model=model,
    args=args,
    train_loader=train_loader,
    eval_loader=val_loader,
    loss_fn=loss_fn,
    compute_metrics=lambda yhat, y: {
        "acc": (yhat.argmax(-1) == y).float().mean().item()
    },
)

trainer.train()
metrics = trainer.evaluate()

import matplotlib.pyplot as plt

history = trainer.state["log_history"]
rows = [h for h in history if "eval_acc" in h]

epochs = [h["epoch"] + 1 for h in rows]
accs = [h["eval_acc"] for h in rows]

plt.figure(figsize=(7, 4))
plt.plot(epochs, accs, color="C0")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.ylim(0.0, 1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(list(zip(epochs, accs)))
