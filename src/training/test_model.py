import torch

from src.training.model import BaselineCNN


def main():
    model = BaselineCNN()

    dummy_input = torch.randn(4, 3, 32, 32)

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
