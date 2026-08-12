from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


MANIFEST_PATH = Path("data/manifests/dataset_manifest.csv")


def create_splits() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)

    train_data = manifest[
        manifest["split"] == "train"
    ].copy()

    test_data = manifest[
        manifest["split"] == "test"
    ].copy()

    train_data, validation_data = train_test_split(
        train_data,
        test_size=0.1,
        random_state=42,
        stratify=train_data["label_id"],
    )

    train_data["split"] = "train"
    validation_data["split"] = "validation"
    test_data["split"] = "test"

    final_manifest = pd.concat(
        [
            train_data,
            validation_data,
            test_data,
        ],
        ignore_index=True,
    )

    final_manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(validation_data)}")
    print(f"Test: {len(test_data)}")


if __name__ == "__main__":
    create_splits()
