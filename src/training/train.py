from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.chunk_dataset import ChunkDataset
from src.training.model import BaselineCNN
import csv



TRAIN_DIR = "data/processed/train"
VALIDATION_DIR = "data/processed/validation"
TEST_DIR = "data/processed/test"


BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_PATH = Path(
    "reports/training_metrics.csv"
)

METRICS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def train_one_epoch(
    model,
    loader,
    loss_function,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def evaluate(
    model,
    loader,
    loss_function,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_function(
                outputs,
                labels,
            )

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    device = get_device()

    print(f"Using device: {device}")

    train_dataset = ChunkDataset(
        TRAIN_DIR
    )

    validation_dataset = ChunkDataset(
    VALIDATION_DIR
    )


    test_dataset = ChunkDataset(
        TEST_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    )


    model = BaselineCNN(
        num_classes=10
    ).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            loss_function,
            device,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )


        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"checkpoint_epoch_{epoch + 1:02d}.pt"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            },
            checkpoint_path,
        )

        print(f"Checkpoint saved: {checkpoint_path}")



    model_path = MODEL_DIR / "baseline_cnn.pt"

    torch.save(
        model.state_dict(),
        model_path,
    )

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
