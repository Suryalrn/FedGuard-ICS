
import torch

from tcn import FedGuardTCN


def main():
    model = FedGuardTCN(
        input_features=51,
        hidden_channels=64,
        num_classes=2
    )

    # Simulate a batch of 32 SWaT sequences.
    dummy_input = torch.randn(32, 60, 51)

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    print("\nModel parameters:")
    print(sum(p.numel() for p in model.parameters()))

    assert output.shape == (32, 2)

    print("\nTCN architecture test passed!")


if __name__ == "__main__":
    main()
