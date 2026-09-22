import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

from ..mpc_trainer import PCTrainer, PCTrainingArguments
from .model import PCModel

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps"

    T = 10
    batch_size = 256
    all_epoch = 20

    train_dataset = mnist.MNIST(
        root="./data", download=False, train=True, transform=ToTensor()
    )
    test_dataset = mnist.MNIST(
        root="./data", download=False, train=False, transform=ToTensor()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = PCModel().to(device)
    model.train()

    training_args = PCTrainingArguments(
        T=T,
        num_train_epochs=all_epoch,
        optimizer_x_kwargs={"lr": 0.2},
        optimizer_p_cls=SGD,
        # optimizer_p_kwargs={"lr": 1e-1},
        optimizer_p_kwargs={"lr": 1e-1, "weight_decay": 1e-4},
        update_p_at="last",
        eval_strategy="epoch",
        logging_steps=20,
        device=device,
    )

    trainer = PCTrainer(
        model=model,
        args=training_args,
        train_loader=train_loader,
        eval_loader=test_loader,
        loss_fn=CrossEntropyLoss(),
        compute_metrics=lambda yhat, y: {
            "acc": (yhat.argmax(dim=-1) == y).float().mean().item()
        },
    )

    prev_acc = 0
    save_steps = 100

    state = trainer.train()
    metrics = trainer.evaluate()
    print("test:", metrics)

    rows = [h for h in state["log_history"] if "eval_acc" in h]
    for h in rows:
        print(f"epoch {h['epoch']+1}, accuracy {h['eval_acc']}")

    epochs = [h["epoch"] + 1 for h in rows]
    accuracies = [h["eval_acc"] for h in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST LeNet5 Test Accuracy")
    plt.xticks(epochs)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
