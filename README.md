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
- `--epochs`: Training epochs (default `5`). Increase for better accuracy.
- `--device`: Force `mps`, `cuda`, or `cpu`. By default the script picks the best available backend.
- `--skip-train`: Export the randomly initialised weights without running training (useful for debugging the pipeline).

After training, update `CONFIG.weightUrl` in `assets/main-CKMdEyXu.js` if you export to a different location/name. Refresh the browser to load the new weights.

## Notes & Tips

- The visualiser highlights the top-N (configurable) strongest incoming connections per neuron to keep the scene legible.
- Colors encode activation sign and magnitude (cool tones for negative/low, warm tones for strong positive contributions).
- The default export (`exports/sample_mlp_weights.json`) comes from a quick one-epoch training run and should reach ~94% accuracy. Retrain for higher fidelity.
- If you adjust the architecture, ensure the JSON export reflects the new layer sizes; the front-end builds the scene dynamically from that metadata.

## Deployment

The server keeps live assets separate from active development under `releases/`:

- `releases/repo.git` – bare repository that receives pushes.
- `releases/current/` – files served by your static HTTP server.
- `releases/backups/<timestamp>/` – point-in-time snapshots for quick rollback.

Initial setup (already done if you are reading this after cloning from the server):

```bash
mkdir -p releases/current releases/backups releases/.deploy_tmp
git init --bare releases/repo.git
```

At deploy time the `post-receive` hook in `releases/repo.git/hooks/` invokes `deploy.sh`, which:

1. Checks out the pushed commit into `releases/.deploy_tmp`.
2. Syncs those files into `releases/current/`.
3. Copies the same tree into `releases/backups/<timestamp>/` and records the commit hash.

Only pushes to the `production` branch trigger the hook. To ship a new release from your working clone:

```bash
git checkout production
# merge or cherry-pick changes you want to deploy
git push server production
```

Where the `server` remote points at the bare repo on the host:

```bash
git remote add server user@host:/home/Neural-Network-Visualisation/releases/repo.git
```

You can also deploy a specific commit manually:

```bash
./deploy.sh <commit_sha>
```

Running the script without arguments prints a usage message.
