# MNIST MLP Visualizer

Interactive web visualisation for a compact multi-layer perceptron trained on the MNIST handwritten digit dataset. Draw a digit, watch activations propagate through the network in 3D, and inspect real-time prediction probabilities.

## Repository Layout

- `index.html` / `assets/` – Static Three.js visualiser and UI assets.
- `exports/sample_mlp_weights.json` – Default weights exported after a 1-epoch training run.
- `training/mlp_train.py` – PyTorch helper to train the MLP (with Apple Metal acceleration when available) and export weights for the front-end.

## Quick Start

1. **Install Python dependencies** (PyTorch + torchvision):

   ```bash
   python3 -m pip install torch torchvision
   ```

2. **Launch a static file server** from the repository root (any server works; this example uses Python):

   ```bash
   python3 -m http.server 8000
   ```

3. Open `http://localhost:8000` in your browser. Draw on the 28×28 grid (left-click to draw, right-click to erase) and explore the 3D network with the mouse or trackpad.

## Training & Exporting New Weights

`training/mlp_train.py` trains a small MLP on MNIST and writes a JSON export the front-end consumes. Metal (MPS) is used automatically when available on Apple Silicon; otherwise the script falls back to CUDA or CPU.

Typical usage:

```bash
python3 training/mlp_train.py \
  --epochs 5 \
  --hidden-dims 128 64 \
  --batch-size 256 \
  --export-path exports/mlp_weights.json
```

Key options:

- `--hidden-dims`: Hidden layer sizes (default `128 64`). Keep the network modest so the visualisation stays responsive.
- `--epochs`: Minimum training epochs (default `5`). The script will automatically extend the run so the timeline hits the 10× dataset milestone.
- `--batch-size`: Mini-batch size (default `128`).
- `--device`: Force `mps`, `cuda`, or `cpu`. By default the script picks the best available backend.
- `--skip-train`: Export the randomly initialised weights without running training (useful for debugging the pipeline).

After training, update `VISUALIZER_CONFIG.weightUrl` in `assets/main.js` if you export to a different location/name. Refresh the browser to load the new weights.

### Training timeline export

Every exported JSON now includes a `timeline` array capturing the network at key milestones (≈100, 1 000, 3 000, 10 000, 30 000 images plus 1×, 2×, 5×, 10× dataset passes). The slider at the bottom of the UI lets you swap between these snapshots, showing how weights, activations, and accuracy evolve from random initialisation through 10 full passes over MNIST. Re-export the weights with the updated script to generate fresh timeline data for your own runs.

## Notes & Tips

- The visualiser highlights the top-N (configurable) strongest incoming connections per neuron to keep the scene legible.
- Colors encode activation sign and magnitude (cool tones for negative/low, warm tones for strong positive contributions).
- The default export (`exports/sample_mlp_weights.json`) comes from a quick one-epoch training run and should reach ~94% accuracy. Retrain for higher fidelity.
- If you adjust the architecture, ensure the JSON export reflects the new layer sizes; the front-end builds the scene dynamically from that metadata.

## Deployment

The server keeps live assets separate from active development under `releases/`:

- `releases/current/` – files served by your static HTTP server.
- `releases/backups/<timestamp>/` – point-in-time snapshots for quick rollback.
- `releases/.deploy_tmp/` – staging area used during deployment.

To publish the code you currently have checked out, run the deploy script from the repository root:

```bash
./deploy.sh
```

You can target a different commit or branch explicitly:

```bash
./deploy.sh <commit-ish>
```

The script exports the requested commit into the staging area, syncs it into `releases/current/`, and saves the same tree under `releases/backups/<timestamp>/` with the commit hash recorded in `.commit`.
