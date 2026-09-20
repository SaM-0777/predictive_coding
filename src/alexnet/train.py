import os
from pathlib import Path
import shutil
from urllib.request import urlretrieve
import zipfile
from matplotlib import pyplot as plt
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torchvision import transforms, datasets
import pandas as pd
from tqdm import tqdm

from .model import AlexNet

CELEBA_URL = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
DATASET_DIR = Path("data/celeba")


def get_celeba_dataset():
    dataset_dir = DATASET_DIR
    zip_path = dataset_dir / "celeba.zip"
    image_dir = dataset_dir / "data" / "img_align_celeba"

    dataset_dir.mkdir(parents=True, exist_ok=True)

    if image_dir.exists():
        image_count = len(list(image_dir.glob("*.jpg")))

        if image_count > 0:
            print(f"CelebA already exists: {image_count} images")
            return dataset_dir

    if not zip_path.exists():
        print("Downloading CelebA Dataset...")

        urlretrieve(
            CELEBA_URL,
            zip_path,
            reporthook=lambda block_num, block_size, total_size: print(
                f"\rDownloaded: "
                f"{min(block_num * block_size / total_size * 100, 100):.1f}%",
                end="",
            ),
        )

        print("\nDownload complete.")
    else:
        print("CelebA ZIP already exists. Skipping download.")

    print("Extracting CelebA...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir / "data")

    print("Extraction complete.")

    image_count = len(list(image_dir.glob("*.jpg")))
    print(f"Found {image_count} images.")

    return dataset_dir


def get_annotations():
    dataset_dir = DATASET_DIR
    annotations_dir = dataset_dir / "list_attr_celeba.csv"

    df = pd.read_csv(annotations_dir)
    df = df[["image_id", "Smiling"]]
    return df


def arrange_data(df: pd.DataFrame, N: int = 25000):
    dataset_dir = DATASET_DIR / "data"

    smile_dir = dataset_dir / "smile"
    no_smile_dir = dataset_dir / "no_smile"

    smile_dir.mkdir(parents=True, exist_ok=True)
    no_smile_dir.mkdir(parents=True, exist_ok=True)

    existing_smile = list(smile_dir.glob("*.jpg"))
    existing_no_smile = list(no_smile_dir.glob("*.jpg"))

    if existing_smile or existing_no_smile:
        print(
            f"Dataset already arranged. "
            f"Smile: {len(existing_smile)}, "
            f"No smile: {len(existing_no_smile)}"
        )

        img_list = [image.name for image in existing_smile]
        img_list.extend(image.name for image in existing_no_smile)

        return img_list

    for i, (_, i_row) in enumerate(df.iterrows()):
        image_path = dataset_dir / ("img_align_celeba/" + i_row["image_id"])

        if i < N:
            if i_row["Smiling"] == 1:
                destination = smile_dir / i_row["image_id"]
            else:
                destination = no_smile_dir / i_row["image_id"]

            shutil.copyfile(
                image_path,
                destination,
            )
        else:
            break

    img_list = os.listdir(dataset_dir / "smile/")
    img_list.extend(os.listdir(dataset_dir / "no_smile/"))

    return img_list


def save_model_weights(model, path="models/alexnet_smile_weights.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to: {path}")


from torch.utils.data import Dataset
from PIL import Image


class CelebASmileDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform

        self.classes = ["no_smile", "smile"]
        self.class_to_idx = {
            "no_smile": 0,
            "smile": 1,
        }

        self.samples = []

        for class_name in self.classes:
            class_dir = dataset_dir / class_name
            label = self.class_to_idx[class_name]

            for image_path in class_dir.glob("*.jpg"):
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    dataset_dir = get_celeba_dataset()
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

    train, test = torch.utils.data.random_split(celeba_data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    model = AlexNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = CrossEntropyLoss()

    epochs = 10
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for current_epoch in range(epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {current_epoch + 1}/{epochs} [Train]",
        )

        for image, label in train_progress:
            image = image.to(device)
            label = label.to(device).long()
            optimizer.zero_grad()
            pred = model(image)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * image.size(0)
            predicted_class = pred.argmax(dim=1)

            train_correct += (predicted_class == label).sum().item()

            train_total += label.size(0)

            current_loss = train_loss / train_total
            current_accuracy = (train_correct / train_total) * 100

            train_progress.set_postfix(
                loss=f"{current_loss:.4f}",
                acc=f"{current_accuracy:.2f}%",
            )

        epoch_train_loss = train_loss / train_total
        epoch_train_accuracy = (train_correct / train_total) * 100

        model.eval()

        test_loss = 0.0
        test_correct = 0
        test_total = 0

        test_progress = tqdm(
            test_loader,
            desc=f"Epoch {current_epoch + 1}/{epochs} [Test]",
        )

        with torch.no_grad():
            for image, label in test_progress:
                image = image.to(device)
                label = label.to(device).long()
                pred = model(image)
                loss = loss_fn(pred, label)
                test_loss += loss.item() * image.size(0)
                predicted_class = pred.argmax(dim=1)
                test_correct += (predicted_class == label).sum().item()
                test_total += label.size(0)

                current_loss = test_loss / test_total
                current_accuracy = (test_correct / test_total) * 100

                test_progress.set_postfix(
                    loss=f"{current_loss:.4f}",
                    acc=f"{current_accuracy:.2f}%",
                )

        epoch_test_loss = test_loss / test_total
        epoch_test_accuracy = (test_correct / test_total) * 100

        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)

        train_accuracies.append(epoch_train_accuracy)
        test_accuracies.append(epoch_test_accuracy)

        print(f"\nEpoch {current_epoch + 1}/{epochs}")
        print(
            f"Train | "
            f"Loss: {epoch_train_loss:.4f} | "
            f"Accuracy: {epoch_train_accuracy:.2f}%"
        )
        print(
            f"Test  | "
            f"Loss: {epoch_test_loss:.4f} | "
            f"Accuracy: {epoch_test_accuracy:.2f}%"
        )

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs_range,
        train_losses,
        label="Train Loss",
        marker="o",
    )
    plt.plot(
        epochs_range,
        test_losses,
        label="Test Loss",
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.xticks(epochs_range)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs_range,
        train_accuracies,
        label="Train Accuracy",
        marker="o",
    )

    plt.plot(
        epochs_range,
        test_accuracies,
        label="Test Accuracy",
        marker="o",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Test Accuracy")

    plt.xticks(epochs_range)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.show()

    save_model_weights(model, "models/alexnet_smile_weights.pth")
