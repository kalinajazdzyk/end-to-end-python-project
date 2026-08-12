from pathlib import Path

from torchvision.datasets import CIFAR10


DATA_DIR = Path("data/raw")


def download_cifar10() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CIFAR-10...")

    CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
    )

    CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
    )

    print("CIFAR-10 download complete.")


if __name__ == "__main__":
    download_cifar10()
