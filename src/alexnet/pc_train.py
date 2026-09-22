from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms

import matplotlib.pyplot as plt

from .model import PCAlexNet
from .train import CelebASmileDataset, arrange_data, get_annotations
from ..mpc_trainer import PCTrainer, PCTrainingArguments


def accuracy(model, loader, device, mode="eval"):
    model.eval() if mode == "eval" else model.train()
    correct = total = 0
    with torch.no_grad() if mode == "eval" else torch.enable_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if mode == "eval":
                pred = model(x)
            else:
                # one train-mode forward (uses existing x if any — reset first)
                for m in model.modules():
                    if hasattr(m, "set_sample_x"):
                        m.set_sample_x(True)
                pred = model(x)
            correct += (pred.argmax(-1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


if __name__ == "__main__":
    torch.Generator().manual_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    dataset_dir = Path("./data/celeba")
    df = get_annotations()
    img_list = arrange_data(df)

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # celeba_data = datasets.ImageFolder(dataset_dir / "data", transform=transform)
    celeba_data = CelebASmileDataset(
        dataset_dir / "data",
        transform=transform,
    )

    print(f"dataset_dir / 'data'", dataset_dir / "data")
    print(f"classes {celeba_data.classes}")
    print(f"Total len {len(celeba_data)}")

    train_size = int(len(img_list) * 0.75)
    test_size = len(img_list) - train_size

    print(train_size, test_size)

    T = 10  # was 5
    epochs = 10
    batch_size = 32  # was 64

    train, test = torch.utils.data.random_split(celeba_data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    model = PCAlexNet()
    model.to(device)
    model.train()

    training_args = PCTrainingArguments(
        T=T,
        num_train_epochs=epochs,
        # energy_coefficient=1.0,  # only if PCLayer energy is .mean(), its currently NOT mean
        optimizer_x_kwargs={"lr": 0.2},
        optimizer_p_cls=torch.optim.Adam,
        # optimizer_p_kwargs={"lr": 2e-4},
        optimizer_p_kwargs={"lr": 1e-4, "weight_decay": 1e-4},
        update_p_at="last",
        eval_strategy="epoch",
        logging_steps=10,
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

    state = trainer.train()
    metrics = trainer.evaluate()
    print("test:", metrics)

    print("train eval-mode ", accuracy(model, train_loader, device, "eval"))
    print("test  eval-mode ", accuracy(model, test_loader, device, "eval"))
    print("train train-mode", accuracy(model, train_loader, device, "train"))

    torch.save(model, f"models/pc_alexnet_0.pth")
    print(f"Model saved at step")

    rows = [h for h in state["log_history"] if "eval_acc" in h]

    epochs = [h["epoch"] + 1 for h in rows]
    accuracies = [h["eval_acc"] for h in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("PCAlexNet CelebA smile")
    plt.xticks(epochs)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

"""
I am working on a project to implement predictive coding in  ppo

I have attached the PCLayer and PCTrainer class script 

And right now I have implemented a ppo script to let an agent learn catcher gameplay and I hv attached the ppo script here too catcher.py

currently in catcher.py I am using backpropagation but now I want to use predictive coding algorithm instead of back propagation
"""
