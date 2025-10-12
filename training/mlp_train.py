"""Utility for training a small MNIST MLP and exporting weights for the visualization."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def resolve_device(preferred: str | None = None) -> torch.device:
    """Return the best available device, prioritising MPS for Apple silicon."""
    if preferred:
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "cpu":
            return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SmallMLP(nn.Module):
    """Simple fully connected network for MNIST digits."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int = 10):
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(x.size(0), -1)
        return self.net(x)


@dataclass
class ExportLayer:
    """Serializable representation of a dense layer."""

    layer_index: int
    type: str
    name: str
    activation: str
    weight_shape: tuple[int, int]
    bias_shape: tuple[int]
    weights: list[list[float]]
    biases: list[float]


def export_model(model: SmallMLP, output_path: Path, hidden_activations: Sequence[str]) -> None:
    """Export the model parameters to JSON for the front-end."""
    layers: list[ExportLayer] = []
    dense_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    for idx, (layer, activation) in enumerate(zip(dense_layers, hidden_activations, strict=False)):
        weights = layer.weight.detach().cpu().tolist()
        biases = layer.bias.detach().cpu().tolist()
        export_layer = ExportLayer(
            layer_index=idx,
            type="dense",
            name=f"dense_{idx}",
            activation=activation,
            weight_shape=tuple(layer.weight.shape),  # type: ignore[arg-type]
            bias_shape=tuple(layer.bias.shape),  # type: ignore[arg-type]
            weights=weights,
            biases=biases,
        )
        layers.append(export_layer)

    payload: dict[str, Any] = {
        "network": {
            "architecture": [layer.weight_shape[1] for layer in layers] + [layers[-1].weight_shape[0]],
            "layers": [asdict(layer) for layer in layers],
            "input_dim": layers[0].weight_shape[1],
            "output_dim": layers[-1].weight_shape[0],
            "normalization": {"mean": MNIST_MEAN, "std": MNIST_STD},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: optim.Optimizer) -> float:
    model.train()
    running_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def parse_hidden_dims(raw: Sequence[int]) -> list[int]:
    dims = [int(d) for d in raw if int(d) > 0]
    if not dims:
        raise ValueError("At least one positive hidden dimension is required.")
    return dims


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small MNIST MLP and export weights.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden layer sizes, e.g. --hidden-dims 128 64.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader worker processes (set to 0 if you hit spawn issues).",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("exports/mlp_weights.json"),
        help="Where to write the exported weights JSON.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for MNIST downloads.",
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default=None,
        help="Force a specific device (defaults to the best available).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and just export the randomly initialised weights.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    print(f"Using device: {device}")

    model = SmallMLP(28 * 28, hidden_dims).to(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(args.num_workers, 0),
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=max(args.num_workers, 0),
        pin_memory=pin_memory,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if not args.skip_train:
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, device, optimizer)
            acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:02d} - loss: {loss:.4f} - test accuracy: {acc*100:.2f}%")

    # Build activation list: ReLU for every hidden layer, softmax for the output (handled client-side).
    hidden_activations = ["relu"] * len(hidden_dims) + ["linear"]
    export_model(model, args.export_path, hidden_activations)
    print(f"Exported weights to {args.export_path.resolve()}")


if __name__ == "__main__":
    main()
