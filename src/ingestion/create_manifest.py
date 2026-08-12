from pathlib import Path

import pandas as pd
from torchvision.datasets import CIFAR10


DATA_DIR = Path("data/raw")
MANIFEST_DIR = Path("data/manifests")


def create_manifest() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for train, split in [(True, "train"), (False, "test")]:
        dataset = CIFAR10(
            root=DATA_DIR,
            train=train,
            download=False,
        )

        for index, (_, label) in enumerate(dataset):
            rows.append(
                {
                    "index": index,
                    "label_id": label,
                    "label": dataset.classes[label],
                    "split": split,
                }
            )

    manifest = pd.DataFrame(rows)

    output_path = MANIFEST_DIR / "dataset_manifest.csv"
    manifest.to_csv(output_path, index=False)

    print(f"Manifest written to: {output_path}")
    print(f"Total samples: {len(manifest)}")
    print("\nClass distribution:")
    print(manifest["label"].value_counts())


if __name__ == "__main__":
    create_manifest()
