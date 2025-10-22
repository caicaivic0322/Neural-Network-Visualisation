"""Utility for training a small MNIST MLP and exporting weights for the visualization."""
from __future__ import annotations

import argparse
import json
import math
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


@dataclass
class TimelineMilestone:
    """Milestone definition for the training timeline export."""

    identifier: str
    threshold_images: int
    label: str
    kind: str
    dataset_multiple: float | None = None


def export_model(
    model: SmallMLP,
    output_path: Path,
    hidden_activations: Sequence[str],
    timeline: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Export the model parameters and optional training timeline to JSON."""
    layers = create_dense_layer_exports(model, hidden_activations)
    payload: dict[str, Any] = {
        "network": build_network_payload(layers),
    }
    if timeline:
        payload["timeline"] = timeline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


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


def create_dense_layer_exports(model: SmallMLP, activations: Sequence[str]) -> list[ExportLayer]:
    """Convert the model's dense layers into serialisable export payloads."""
    dense_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    layers: list[ExportLayer] = []
    for idx, (layer, activation) in enumerate(zip(dense_layers, activations, strict=False)):
        layers.append(
            ExportLayer(
                layer_index=idx,
                type="dense",
                name=f"dense_{idx}",
                activation=activation,
                weight_shape=tuple(layer.weight.shape),  # type: ignore[arg-type]
                bias_shape=tuple(layer.bias.shape),  # type: ignore[arg-type]
                weights=layer.weight.detach().cpu().tolist(),
                biases=layer.bias.detach().cpu().tolist(),
            )
        )
    return layers


def build_network_payload(layers: Sequence[ExportLayer]) -> dict[str, Any]:
    return {
        "architecture": [layer.weight_shape[1] for layer in layers] + [layers[-1].weight_shape[0]],
        "layers": [asdict(layer) for layer in layers],
        "input_dim": layers[0].weight_shape[1],
        "output_dim": layers[-1].weight_shape[0],
        "normalization": {"mean": MNIST_MEAN, "std": MNIST_STD},
    }


def build_default_timeline(dataset_size: int) -> list[TimelineMilestone]:
    approx_spec = [
        TimelineMilestone("initial", 0, "Initial weights", "initial"),
        TimelineMilestone("approx_100", 100, "≈100 images", "approx"),
        TimelineMilestone("approx_1k", 1_000, "≈1k images", "approx"),
        TimelineMilestone("approx_3k", 3_000, "≈3k images", "approx"),
        TimelineMilestone("approx_10k", 10_000, "≈10k images", "approx"),
        TimelineMilestone("approx_30k", 30_000, "≈30k images", "approx"),
    ]
    multiples = [
        TimelineMilestone("dataset_1x", dataset_size, "1× dataset", "dataset_multiple", 1.0),
        TimelineMilestone("dataset_2x", dataset_size * 2, "2× dataset", "dataset_multiple", 2.0),
        TimelineMilestone("dataset_5x", dataset_size * 5, "5× dataset", "dataset_multiple", 5.0),
        TimelineMilestone("dataset_10x", dataset_size * 10, "10× dataset", "dataset_multiple", 10.0),
    ]
    milestones = approx_spec + multiples
    milestones.sort(key=lambda item: item.threshold_images)
    return milestones


def format_snapshot_description(
    milestone: TimelineMilestone,
    images_seen: int,
    batches_seen: int,
    dataset_size: int,
) -> str:
    if milestone.kind == "initial":
        return "0 images processed (random initialisation)"
    dataset_passes = images_seen / dataset_size if dataset_size else 0.0
    human_images = f"{images_seen:,}"
    batches = f"{batches_seen:,}"
    if milestone.kind == "dataset_multiple" and milestone.dataset_multiple is not None:
        multiplier = f"{milestone.dataset_multiple:g}× dataset"
        return f"{human_images} images • {multiplier} • {batches} batches"
    return f"{human_images} images processed • {batches} batches • {dataset_passes:.2f}× dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small MNIST MLP and export weights.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Minimum number of epochs. The run extends automatically to reach all timeline milestones.",
    )
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
    dataset_size = len(train_dataset)
    milestones = build_default_timeline(dataset_size)
    if not milestones:
        raise RuntimeError("No timeline milestones defined.")

    # Build activation list: ReLU for every hidden layer, softmax for the output (handled client-side).
    hidden_activations = ["relu"] * len(hidden_dims) + ["linear"]

    timeline_entries: list[dict[str, Any]] = []
    cumulative_loss = 0.0
    images_seen = 0
    global_step = 0
    milestone_index = 0
    last_eval_accuracy = 0.0
    total_required_images = milestones[-1].threshold_images
    required_epochs = math.ceil(total_required_images / dataset_size) if dataset_size else 0
    target_epochs = max(args.epochs, required_epochs)

    def record_snapshot(milestone: TimelineMilestone) -> None:
        nonlocal last_eval_accuracy
        accuracy = evaluate(model, test_loader, device)
        last_eval_accuracy = accuracy
        layers = create_dense_layer_exports(model, hidden_activations)
        entry: dict[str, Any] = {
            "id": milestone.identifier,
            "order": len(timeline_entries),
            "label": milestone.label,
            "kind": milestone.kind,
            "target_images": milestone.threshold_images,
            "images_seen": images_seen,
            "batches_seen": global_step,
            "dataset_passes": images_seen / dataset_size if dataset_size else 0.0,
            "description": format_snapshot_description(milestone, images_seen, global_step, dataset_size),
            "metrics": {
                "test_accuracy": accuracy,
            },
            "layers": [asdict(layer) for layer in layers],
        }
        if milestone.dataset_multiple is not None:
            entry["dataset_multiple"] = milestone.dataset_multiple
        if images_seen > 0:
            entry["metrics"]["avg_training_loss"] = cumulative_loss / images_seen
        timeline_entries.append(entry)
        print(
            f"[Timeline] Captured '{milestone.label}' at {images_seen:,} images "
            f"({global_step:,} batches) – accuracy: {accuracy * 100:.2f}%"
        )

    def advance_milestones() -> None:
        nonlocal milestone_index
        while milestone_index < len(milestones) and images_seen >= milestones[milestone_index].threshold_images:
            record_snapshot(milestones[milestone_index])
            milestone_index += 1

    advance_milestones()

    if not args.skip_train:
        training_complete = milestone_index >= len(milestones)
        for epoch in range(1, target_epochs + 1):
            if training_complete:
                break
            model.train()
            epoch_loss = 0.0
            epoch_images = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                batch_size = data.size(0)
                images_seen += batch_size
                global_step += 1
                cumulative_loss += loss.item() * batch_size
                epoch_loss += loss.item() * batch_size
                epoch_images += batch_size

                advance_milestones()
                if milestone_index >= len(milestones):
                    training_complete = True
                    break

            avg_epoch_loss = epoch_loss / epoch_images if epoch_images else 0.0
            if not training_complete:
                # Ensure we keep tabs on accuracy even if no milestone was reached in this epoch.
                last_eval_accuracy = evaluate(model, test_loader, device)
            print(
                f"Epoch {epoch:02d} - avg loss: {avg_epoch_loss:.4f} - "
                f"test accuracy: {last_eval_accuracy * 100:.2f}% - "
                f"images seen: {images_seen:,}"
            )

    if not timeline_entries:
        # If training was skipped, at least export the initial snapshot.
        record_snapshot(milestones[0])

    export_model(model, args.export_path, hidden_activations, timeline_entries)
    print(f"Exported weights to {args.export_path.resolve()}")


if __name__ == "__main__":
    main()
