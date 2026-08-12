from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChunkDataset(Dataset):
    """PyTorch Dataset that reads processed .npz chunks."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

        self.chunk_paths = sorted(
            self.data_dir.glob("chunk_*.npz")
        )

        if not self.chunk_paths:
            raise FileNotFoundError(
                f"No chunks found in {self.data_dir}"
            )

        # Build an index mapping:
        # global sample index -> (chunk, position)
        self.index = []

        for chunk_id, chunk_path in enumerate(self.chunk_paths):
            with np.load(chunk_path) as data:
                num_samples = len(data["labels"])

            for position in range(num_samples):
                self.index.append(
                    (chunk_id, position)
                )

        print(
            f"Found {len(self.chunk_paths)} chunks "
            f"containing {len(self.index)} samples."
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        chunk_id, position = self.index[idx]

        chunk_path = self.chunk_paths[chunk_id]

        with np.load(chunk_path) as data:
            image = data["images"][position]
            label = data["labels"][position]

        # NumPy HWC → PyTorch CHW
        image = torch.from_numpy(image).permute(2, 0, 1)

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image, label
