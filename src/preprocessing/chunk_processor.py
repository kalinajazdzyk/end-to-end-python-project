from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import CIFAR10


DATA_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifests/dataset_manifest.csv")
OUTPUT_DIR = Path("data/processed")

CHUNK_SIZE = 1000
IMAGE_SIZE = (32, 32)


def load_image(dataset, index: int) -> np.ndarray:
    image, _ = dataset[index]

    image = image.resize(IMAGE_SIZE)
    image = np.asarray(image, dtype=np.float32)

    # Normalize pixels from [0, 255] to [0, 1]
    image /= 255.0

    return image


def process_chunk(
    dataset,
    chunk: pd.DataFrame,
    chunk_id: int,
    split: str,
) -> None:
    images = []
    labels = []

    for _, row in chunk.iterrows():
        image = load_image(dataset, int(row["index"]))

        images.append(image)
        labels.append(int(row["label_id"]))

    images_array = np.stack(images)
    labels_array = np.asarray(labels)

    output_dir = OUTPUT_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"chunk_{chunk_id:04d}.npz"

    np.savez_compressed(
        output_path,
        images=images_array,
        labels=labels_array,
    )

    print(
        f"Processed {split} chunk {chunk_id}: "
        f"{len(images_array)} images → {output_path}"
    )


def process_split(
    manifest: pd.DataFrame,
    split: str,
) -> None:
    split_manifest = manifest[
        manifest["split"] == split
    ].reset_index(drop=True)

    dataset = CIFAR10(
        root=DATA_DIR,
        train=(split == "train"),
        download=False,
    )

    total = len(split_manifest)

    for chunk_id, start in enumerate(
        range(0, total, CHUNK_SIZE)
    ):
        end = min(start + CHUNK_SIZE, total)

        chunk = split_manifest.iloc[start:end]

        process_chunk(
            dataset=dataset,
            chunk=chunk,
            chunk_id=chunk_id,
            split=split,
        )


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)

    print(f"Loaded manifest: {len(manifest)} samples")

    process_split(manifest, "train")
    process_split(manifest, "test")

    print("Chunk processing complete.")


if __name__ == "__main__":
    main()
