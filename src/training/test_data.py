from src.training.chunk_dataset import ChunkDataset


def main():
    dataset = ChunkDataset(
        "data/processed/train"
    )

    print("Dataset length:", len(dataset))

    image, label = dataset[0]

    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Label:", label)


if __name__ == "__main__":
    main()
