import torch
from torch.utils.data import DataLoader

from src.training.chunk_dataset import ChunkDataset


def main():
    dataset = ChunkDataset(
        "data/processed/train"
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    print("Images:", images.shape)
    print("Labels:", labels.shape)


if __name__ == "__main__":
    main()
