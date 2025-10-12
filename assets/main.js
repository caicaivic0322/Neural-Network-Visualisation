import { VisualizerConfig, VisualizerLayout } from "./config.js";
import { softmax } from "./math-utils.js";
import { fetchNetworkConfig, displayError } from "./network-service.js";
import { DrawingGrid } from "./drawing-grid.js";
import { MLPNetwork } from "./mlp-network.js";
import { PredictionChart } from "./prediction-chart.js";
import { NetworkVisualizer } from "./network-visualizer.js";

export async function bootstrapVisualizer() {
  setupInfoModal();

  const definition = await fetchNetworkConfig(VisualizerConfig.weightSource);
  if (!definition?.network) {
    throw new Error("Invalid network definition.");
  }

  const mlp = new MLPNetwork(definition.network);
  const drawingGrid = new DrawingGrid(document.getElementById("gridContainer"), 28, 28, {
    brush: VisualizerConfig.brush,
  });
  const predictionChart = new PredictionChart(document.getElementById("predictionChart"));
  const visualizer = new NetworkVisualizer(mlp, VisualizerLayout);

  const resetBtn = document.getElementById("resetBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      drawingGrid.clear();
      updateNetwork();
    });
  }

  function updateNetwork() {
    const rawInput = drawingGrid.getPixels();
    const forward = mlp.forward(rawInput);
    visualizer.update(forward.activations);

    const logitsTyped =
      forward.preActivations.length > 0
        ? forward.preActivations[forward.preActivations.length - 1]
        : new Float32Array(0);
    const probabilities = softmax(Array.from(logitsTyped));
    predictionChart.update(probabilities);
  }

  drawingGrid.setChangeHandler(() => updateNetwork());
  updateNetwork();
}

function setupInfoModal() {
  const infoButton = document.getElementById("infoButton");
  const infoModal = document.getElementById("infoModal");
  const closeButton = document.getElementById("closeInfoModal");
  if (!infoModal) return;

  const showModal = () => infoModal.classList.add("visible");
  const hideModal = () => infoModal.classList.remove("visible");

  infoButton?.addEventListener("click", showModal);
  closeButton?.addEventListener("click", hideModal);
  infoModal.addEventListener("click", (event) => {
    if (event.target === infoModal) {
      hideModal();
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bootstrapVisualizer().catch((error) => {
    console.error(error);
    displayError("Unable to initialise the visualisation. See console for details.");
  });
});
