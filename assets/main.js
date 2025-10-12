const VISUALIZER_CONFIG = {
  weightUrl: "./exports/sample_mlp_weights.json",
  maxConnectionsPerNeuron: 24,
  layerSpacing: 5.5,
  inputSpacing: 0.24,
  hiddenSpacing: 0.95,
  inputNodeSize: 0.18,
  hiddenNodeRadius: 0.22,
  connectionRadius: 0.017,
  brush: {
    drawRadius: 2.0,
    eraseRadius: 2.5,
    drawStrength: 0.95,
    eraseStrength: 0.95,
    softness: 0.3,
  },
};

document.addEventListener("DOMContentLoaded", () => {
  initializeVisualizer().catch((error) => {
    console.error(error);
    renderErrorMessage("Visualisierung konnte nicht initialisiert werden. Details finden Sie in der Konsole.");
  });
});

async function initializeVisualizer() {
  initializeInfoDialog();

  const definition = await fetchNetworkDefinition(VISUALIZER_CONFIG.weightUrl);
  if (!definition?.network) {
    throw new Error("Ungültige Netzwerkdefinition.");
  }

  const neuralModel = new FeedForwardModel(definition.network);
  const digitCanvas = new DigitSketchPad(document.getElementById("gridContainer"), 28, 28, {
    brush: VISUALIZER_CONFIG.brush,
  });
  const probabilityPanel = new ProbabilityPanel(document.getElementById("predictionChart"));
  const neuralScene = new NeuralVisualizer(neuralModel, {
    layerSpacing: VISUALIZER_CONFIG.layerSpacing,
    maxConnectionsPerNeuron: VISUALIZER_CONFIG.maxConnectionsPerNeuron,
    inputSpacing: VISUALIZER_CONFIG.inputSpacing,
    hiddenSpacing: VISUALIZER_CONFIG.hiddenSpacing,
    inputNodeSize: VISUALIZER_CONFIG.inputNodeSize,
    hiddenNodeRadius: VISUALIZER_CONFIG.hiddenNodeRadius,
    connectionRadius: VISUALIZER_CONFIG.connectionRadius,
  });

  if (typeof window !== "undefined") {
    // Expose scene instance for interactive debugging in DevTools.
    window.neuralScene = neuralScene;
  }

  const resetBtn = document.getElementById("resetBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      digitCanvas.clear();
      refreshNetworkState();
    });
  }

  function refreshNetworkState() {
    const rawInput = digitCanvas.getPixels();
    const propagation = neuralModel.propagate(rawInput);
    const activationsForVisualization = propagation.activations.slice();
    activationsForVisualization[0] = rawInput;
    neuralScene.update(activationsForVisualization, propagation.activations);

    const logitsTyped =
      propagation.preActivations.length > 0
        ? propagation.preActivations[propagation.preActivations.length - 1]
        : new Float32Array(0);
    const probabilities = softmax(Array.from(logitsTyped));
    probabilityPanel.update(probabilities);

    if (typeof window !== "undefined") {
      const scene = window.neuralScene;
      const outputLayer = scene?.layerMeshes?.[scene.layerMeshes.length - 1];
      console.log("Output layer colors", outputLayer?.mesh?.instanceColor?.array);
    }
  }

  digitCanvas.setChangeHandler(() => refreshNetworkState());
  refreshNetworkState();
}

function initializeInfoDialog() {
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

async function fetchNetworkDefinition(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Netzwerkgewichte konnten nicht geladen werden (${response.status})`);
  }
  return response.json();
}

function renderErrorMessage(message) {
  const chart = document.getElementById("predictionChart");
  if (chart) {
    chart.innerHTML = `<p class="error-text">${message}</p>`;
  }
}

class DigitSketchPad {
  constructor(container, rows, cols, options = {}) {
    if (!container) {
      throw new Error("Raster-Container nicht gefunden.");
    }
    this.container = container;
    this.rows = rows;
    this.cols = cols;
    this.values = new Float32Array(rows * cols);
    this.cells = [];
    this.isDrawing = false;
    this.activeMode = "draw";
    this.onChange = null;
    this.pendingChange = false;
    const defaultBrush = {
      drawRadius: 1.2,
      eraseRadius: 1.2,
      drawStrength: 0.85,
      eraseStrength: 0.8,
      softness: 0.5,
    };
    this.brush = Object.assign(defaultBrush, options.brush || {});
    this.buildGrid();
  }

  buildGrid() {
    this.gridElement = document.createElement("div");
    this.gridElement.className = "grid";
    this.gridElement.style.gridTemplateColumns = `repeat(${this.cols}, 1fr)`;
    this.gridElement.style.gridTemplateRows = `repeat(${this.rows}, 1fr)`;

    for (let i = 0; i < this.values.length; i += 1) {
      const cell = document.createElement("div");
      cell.className = "grid-cell";
      cell.dataset.index = String(i);
      this.gridElement.appendChild(cell);
      this.cells.push(cell);
    }

    this.container.innerHTML = "";
    const title = document.createElement("div");
    title.className = "grid-title";
    title.textContent = "Ziffer zeichnen";
    this.container.appendChild(title);
    this.container.appendChild(this.gridElement);

    this.gridElement.addEventListener("pointerdown", (event) => this.handlePointerDown(event));
    this.gridElement.addEventListener("pointermove", (event) => this.handlePointerMove(event));
    window.addEventListener("pointerup", () => this.handlePointerUp());
    this.gridElement.addEventListener("contextmenu", (event) => event.preventDefault());
  }

  setChangeHandler(handler) {
    this.onChange = handler;
  }

  handlePointerDown(event) {
    event.preventDefault();
    const isErase = event.button === 2 || event.buttons === 2;
    this.activeMode = isErase ? "erase" : "draw";
    this.isDrawing = true;
    this.applyPointer(event);
  }

  handlePointerMove(event) {
    if (!this.isDrawing) return;
    this.applyPointer(event);
  }

  handlePointerUp() {
    this.isDrawing = false;
  }

  applyPointer(event) {
    const element = document.elementFromPoint(event.clientX, event.clientY);
    if (!element) return;
    const cell = element.closest("[data-index]");
    if (!cell) return;
    const index = Number(cell.dataset.index);
    if (Number.isNaN(index)) return;
    this.paintCell(index, this.activeMode === "erase");
  }

  paintCell(index, erase = false) {
    const row = Math.floor(index / this.cols);
    const col = index % this.cols;
    if (row < 0 || col < 0) return;
    const changed = this.applyBrush(row, col, erase);
    if (changed) {
      this.scheduleChange();
    }
  }

  applyBrush(centerRow, centerCol, erase = false) {
    const radius = erase ? this.brush.eraseRadius : this.brush.drawRadius;
    const strength = erase ? -this.brush.eraseStrength : this.brush.drawStrength;
    const softness = clamp(this.brush.softness ?? 0.5, 0, 0.95);
    const span = Math.ceil(radius);
    let modified = false;
    for (let row = centerRow - span; row <= centerRow + span; row += 1) {
      if (row < 0 || row >= this.rows) continue;
      for (let col = centerCol - span; col <= centerCol + span; col += 1) {
        if (col < 0 || col >= this.cols) continue;
        const distance = Math.hypot(row - centerRow, col - centerCol);
        if (distance > radius) continue;
        const falloff = 1 - distance / radius;
        if (falloff <= 0) continue;
        const influence = Math.pow(falloff, 1 + softness * 2);
        const delta = strength * influence;
        if (Math.abs(delta) < 1e-3) continue;
        const cellIndex = row * this.cols + col;
        const current = this.values[cellIndex];
        const nextValue = clamp(current + delta, 0, 1);
        if (nextValue === current) continue;
        this.values[cellIndex] = nextValue;
        this.updateCellVisual(cellIndex);
        modified = true;
      }
    }
    return modified;
  }

  updateCellVisual(index) {
    const cell = this.cells[index];
    if (!cell) return;
    const value = this.values[index];
    if (value <= 0) {
      cell.style.background = "rgba(255, 255, 255, 0.05)";
      cell.classList.remove("active");
      return;
    }
    const hue = 180 - value * 70;
    const saturation = 70 + value * 25;
    const lightness = 25 + value * 40;
    cell.style.background = `hsl(${hue.toFixed(0)}, ${saturation.toFixed(0)}%, ${lightness.toFixed(0)}%)`;
    cell.classList.add("active");
  }

  scheduleChange() {
    if (this.pendingChange) return;
    this.pendingChange = true;
    requestAnimationFrame(() => {
      this.pendingChange = false;
      if (typeof this.onChange === "function") {
        this.onChange();
      }
    });
  }

  getPixels() {
    return Float32Array.from(this.values);
  }

  clear() {
    this.values.fill(0);
    for (let i = 0; i < this.cells.length; i += 1) {
      this.updateCellVisual(i);
    }
    if (typeof this.onChange === "function") {
      this.onChange();
    }
  }
}

class FeedForwardModel {
  constructor(definition) {
    if (!definition.layers?.length) {
      throw new Error("Die Netzwerkdefinition muss Schichten enthalten.");
    }
    this.normalization = definition.normalization ?? { mean: 0, std: 1 };
    this.architecture = Array.isArray(definition.architecture)
      ? definition.architecture.slice()
      : this.computeArchitecture(definition.layers);
    this.layers = definition.layers.map((layer, index) => ({
      name: layer.name ?? `dense_${index}`,
      activation: layer.activation ?? "relu",
      weights: layer.weights.map((row) => Float32Array.from(row)),
      biases: Float32Array.from(layer.biases),
    }));
  }

  computeArchitecture(layers) {
    if (!layers.length) return [];
    const architecture = [];
    const firstLayer = layers[0];
    architecture.push(firstLayer.weights[0]?.length ?? 0);
    for (const layer of layers) {
      architecture.push(layer.biases.length);
    }
    return architecture;
  }

  propagate(pixels) {
    const { mean, std } = this.normalization;
    const input = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i += 1) {
      input[i] = (pixels[i] - mean) / std;
    }

    const activations = [input];
    const preActivations = [];
    let current = input;

    for (const layer of this.layers) {
      const outSize = layer.biases.length;
      const linear = new Float32Array(outSize);

      for (let neuron = 0; neuron < outSize; neuron += 1) {
        let sum = layer.biases[neuron];
        const weights = layer.weights[neuron];
        for (let source = 0; source < weights.length; source += 1) {
          sum += weights[source] * current[source];
        }
        linear[neuron] = sum;
      }

      preActivations.push(linear);
      let activated;
      if (layer.activation === "relu") {
        activated = new Float32Array(outSize);
        for (let i = 0; i < outSize; i += 1) {
          activated[i] = linear[i] > 0 ? linear[i] : 0;
        }
      } else {
        activated = linear.slice();
      }
      activations.push(activated);
      current = activated;
    }

    return {
      normalizedInput: activations[0],
      activations,
      preActivations,
    };
  }
}

class ProbabilityPanel {
  constructor(container) {
    this.container = container;
    this.rows = [];
    if (!this.container) {
      throw new Error("Vorhersage-Diagrammcontainer nicht gefunden.");
    }
    this.build();
  }

  build() {
    this.container.innerHTML = "";
    const title = document.createElement("h3");
    title.textContent = "Wahrscheinlichkeiten der Ziffern";
    this.container.appendChild(title);

    this.chartElement = document.createElement("div");
    this.chartElement.className = "prediction-chart";
    this.container.appendChild(this.chartElement);

    for (let digit = 0; digit < 10; digit += 1) {
      const row = document.createElement("div");
      row.className = "prediction-bar-container";

      const label = document.createElement("span");
      label.className = "prediction-label";
      label.textContent = String(digit);

      const track = document.createElement("div");
      track.className = "prediction-bar-track";

      const bar = document.createElement("div");
      bar.className = "prediction-bar";
      track.appendChild(bar);

      const value = document.createElement("span");
      value.className = "prediction-percentage";
      value.textContent = "0.0%";

      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(value);
      this.chartElement.appendChild(row);
      this.rows.push({ bar, value });
    }
  }

  update(probabilities) {
    if (!probabilities.length) return;
    const maxProb = Math.max(...probabilities);
    probabilities.forEach((prob, index) => {
      const clamped = Math.max(0, Math.min(1, prob));
      const entry = this.rows[index];
      if (!entry) return;
      entry.bar.style.width = `${(clamped * 100).toFixed(1)}%`;
      entry.value.textContent = `${(clamped * 100).toFixed(1)}%`;
      if (clamped === maxProb) {
        entry.bar.classList.add("highest");
      } else {
        entry.bar.classList.remove("highest");
      }
    });
  }
}

class NeuralVisualizer {
  constructor(mlp, options) {
    this.mlp = mlp;
    this.options = Object.assign(
      {
        layerSpacing: 5.5,
        inputSpacing: 0.24,
        hiddenSpacing: 0.95,
        inputNodeSize: 0.18,
        hiddenNodeRadius: 0.22,
        maxConnectionsPerNeuron: 24,
        connectionRadius: 0.017,
      },
      options || {},
    );
    this.layerMeshes = [];
    this.connectionGroups = [];
    this.tempObject = new THREE.Object3D();
    this.tempColor = new THREE.Color();
    this.initThreeScene();
    this.buildLayers();
    this.buildConnections();
    this.animate();
  }

  initThreeScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 200);
    this.camera.position.set(-15, 0, 15);

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 52;
    this.controls.target.set(0, 0, 0);

    const ambient = new THREE.AmbientLight(0xffffff, 1.2);
    this.scene.add(ambient);
    const hemisphere = new THREE.HemisphereLight(0xffffff, 0x1a1d2e, 0.9);
    hemisphere.position.set(0, 20, 0);
    this.scene.add(hemisphere);
    const directional = new THREE.DirectionalLight(0xffffff, 1.4);
    directional.position.set(18, 26, 24);
    directional.castShadow = true;
    this.scene.add(directional);
    const fillLight = new THREE.DirectionalLight(0xa8c5ff, 0.8);
    fillLight.position.set(-20, 18, -18);
    this.scene.add(fillLight);
    const rimLight = new THREE.PointLight(0x88a4ff, 0.6, 60, 1.6);
    rimLight.position.set(0, 12, -24);
    this.scene.add(rimLight);

    window.addEventListener("resize", () => this.handleResize());
  }

  buildLayers() {
    const inputGeometry = new THREE.BoxGeometry(
      this.options.inputNodeSize,
      this.options.inputNodeSize,
      this.options.inputNodeSize,
    );
    const hiddenGeometry = new THREE.SphereGeometry(this.options.hiddenNodeRadius, 16, 16);
    // Test with MeshBasicMaterial for hidden/output neurons (no lighting influence)
    const hiddenBaseMaterial = new THREE.MeshBasicMaterial();
    hiddenBaseMaterial.toneMapped = false;

    const layerCount = this.mlp.architecture.length;
    const totalWidth = (layerCount - 1) * this.options.layerSpacing;
    const startX = -totalWidth / 2;

    this.mlp.architecture.forEach((neuronCount, layerIndex) => {
      const layerX = startX + layerIndex * this.options.layerSpacing;
      const positions = this.computeLayerPositions(layerIndex, neuronCount, layerX);

      if (layerIndex === 0) {
        const material = new THREE.MeshLambertMaterial();
        material.emissive.setRGB(0.08, 0.08, 0.08);
        const mesh = new THREE.InstancedMesh(inputGeometry, material, neuronCount);
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(neuronCount * 3), 3);
        colorAttribute.setUsage(THREE.DynamicDrawUsage);
        mesh.instanceColor = colorAttribute;

        positions.forEach((position, instanceIndex) => {
          this.tempObject.position.copy(position);
          this.tempObject.updateMatrix();
          mesh.setMatrixAt(instanceIndex, this.tempObject.matrix);
          mesh.setColorAt(instanceIndex, this.tempColor.setRGB(0.15, 0.15, 0.15));
        });

        mesh.instanceMatrix.needsUpdate = true;
        mesh.instanceColor.needsUpdate = true;
        this.scene.add(mesh);
        this.layerMeshes.push({ mesh, positions, type: "input" });
      } else {
        const material = hiddenBaseMaterial.clone();
        // Clone geometry per mesh so each InstancedMesh can have its own instanceColor attribute
        const geometry = hiddenGeometry.clone();
        const mesh = new THREE.InstancedMesh(geometry, material, neuronCount);
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(neuronCount * 3), 3);
        colorAttribute.setUsage(THREE.DynamicDrawUsage);
        mesh.instanceColor = colorAttribute;

        positions.forEach((position, instanceIndex) => {
          this.tempObject.position.copy(position);
          this.tempObject.updateMatrix();
          mesh.setMatrixAt(instanceIndex, this.tempObject.matrix);
          mesh.setColorAt(instanceIndex, this.tempColor.setRGB(0.15, 0.15, 0.15));
        });

        mesh.instanceMatrix.needsUpdate = true;
        mesh.instanceColor.needsUpdate = true;
        this.scene.add(mesh);
        this.layerMeshes.push({ mesh, positions, type: "hidden" });
      }
    });
  }

  computeLayerPositions(layerIndex, neuronCount, layerX) {
    const positions = [];
    if (layerIndex === 0) {
      const spacing = this.options.inputSpacing;
      let rows;
      let cols;
      if (neuronCount === 28 * 28) {
        rows = 28;
        cols = 28;
      } else {
        cols = Math.ceil(Math.sqrt(neuronCount));
        rows = Math.ceil(neuronCount / cols);
      }
      const height = (rows - 1) * spacing;
      const width = (cols - 1) * spacing;
      let filled = 0;
      for (let row = 0; row < rows && filled < neuronCount; row += 1) {
        for (let col = 0; col < cols && filled < neuronCount; col += 1) {
          const y = height / 2 - row * spacing;
          const z = -width / 2 + col * spacing;
          positions.push(new THREE.Vector3(layerX, y, z));
          filled += 1;
        }
      }
    } else {
      const spacing = this.options.hiddenSpacing;
      const cols = Math.max(1, Math.ceil(Math.sqrt(neuronCount)));
      const rows = Math.ceil(neuronCount / cols);
      const height = (rows - 1) * spacing;
      const width = (cols - 1) * spacing;
      for (let index = 0; index < neuronCount; index += 1) {
        const row = Math.floor(index / cols);
        const col = index % cols;
        const y = height / 2 - row * spacing;
        const z = -width / 2 + col * spacing;
        positions.push(new THREE.Vector3(layerX, y, z));
      }
    }
    return positions;
  }

  buildConnections() {
    const connectionRadius = this.options.connectionRadius ?? 0.02;
    const baseGeometry = new THREE.CylinderGeometry(connectionRadius, connectionRadius, 1, 10, 1, true);
    const material = new THREE.MeshLambertMaterial({
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
    });
    // Do not set vertexColors explicitly; instancing color works independently

    this.mlp.layers.forEach((layer, layerIndex) => {
      const { selected, maxAbsWeight } = this.findImportantConnections(layer);
      if (!selected.length) return;

      // Clone geometry per mesh so instanceColor can be bound independently
      const mesh = new THREE.InstancedMesh(baseGeometry.clone(), material.clone(), selected.length);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(selected.length * 3), 3);
      colorAttribute.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = colorAttribute;

      selected.forEach((connection, instanceIndex) => {
        const sourcePosition = this.layerMeshes[layerIndex].positions[connection.sourceIndex];
        const targetPosition = this.layerMeshes[layerIndex + 1].positions[connection.targetIndex];
        const direction = targetPosition.clone().sub(sourcePosition);
        const length = direction.length();
        const midpoint = sourcePosition.clone().addScaledVector(direction, 0.5);

        this.tempObject.position.copy(midpoint);
        const quaternion = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          direction.clone().normalize(),
        );
        this.tempObject.scale.set(1, length, 1);
        this.tempObject.quaternion.copy(quaternion);
        this.tempObject.updateMatrix();
        mesh.setMatrixAt(instanceIndex, this.tempObject.matrix);
        mesh.setColorAt(instanceIndex, this.tempColor.setRGB(1, 1, 1));
      });

      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor.needsUpdate = true;
      this.scene.add(mesh);
      this.connectionGroups.push({
        mesh,
        connections: selected,
        sourceLayer: layerIndex,
        maxAbsWeight,
      });
    });
  }

  findImportantConnections(layer) {
    const limit = this.options.maxConnectionsPerNeuron;
    const selected = [];
    let maxAbsWeight = 0;
    for (let target = 0; target < layer.weights.length; target += 1) {
      const row = layer.weights[target];
      const candidates = [];
      for (let source = 0; source < row.length; source += 1) {
        const weight = row[source];
        if (!Number.isFinite(weight)) continue;
        const magnitude = Math.abs(weight);
        candidates.push({ sourceIndex: source, targetIndex: target, weight, magnitude });
        if (magnitude > maxAbsWeight) maxAbsWeight = magnitude;
      }
      candidates.sort((a, b) => b.magnitude - a.magnitude);
      const take = Math.min(limit, candidates.length);
      for (let i = 0; i < take; i += 1) {
        selected.push({
          sourceIndex: candidates[i].sourceIndex,
          targetIndex: candidates[i].targetIndex,
          weight: candidates[i].weight,
        });
      }
    }
    return { selected, maxAbsWeight };
  }

  update(displayActivations, networkActivations = displayActivations) {
    this.layerMeshes.forEach((layer, layerIndex) => {
      const values = displayActivations[layerIndex];
      if (!values) return;
      const scale = layerIndex === 0 ? 1 : maxAbsValue(displayActivations[layerIndex]);
      this.applyNodeColors(layer, values, scale || 1);
    });

    this.connectionGroups.forEach((group) => {
      const sourceValues = networkActivations[group.sourceLayer];
      if (!sourceValues) return;
      this.applyConnectionColors(group, sourceValues);
    });
  }

  applyNodeColors(layer, values, scale) {
    const { mesh, type } = layer;
    if (type === "input") {
      for (let i = 0; i < values.length; i += 1) {
        const value = clamp(values[i], 0, 1);
        this.tempColor.setRGB(value, value, value);
        mesh.setColorAt(i, this.tempColor);
      }
      mesh.instanceColor.needsUpdate = true;
      return;
    }

    const safeScale = scale > 1e-6 ? scale : 1;
    for (let i = 0; i < values.length; i += 1) {
      const value = values[i];
      const normalized = clamp(value / safeScale, 0, 1);
      this.tempColor.setRGB(normalized, normalized, normalized);
      mesh.setColorAt(i, this.tempColor);
    }
    mesh.instanceColor.needsUpdate = true;
  }

  applyConnectionColors(group, sourceValues) {
    const contributions = new Float32Array(group.connections.length);
    let maxContribution = 0;
    group.connections.forEach((connection, index) => {
      const activation = sourceValues[connection.sourceIndex] ?? 0;
      const contribution = activation * connection.weight;
      contributions[index] = contribution;
      const magnitude = Math.abs(contribution);
      if (magnitude > maxContribution) maxContribution = magnitude;
    });
    const scale = maxContribution > 1e-6 ? maxContribution : group.maxAbsWeight || 1;
    group.connections.forEach((connection, index) => {
      const normalized = clamp(contributions[index] / scale, -1, 1);
      const magnitude = Math.abs(normalized);
      if (magnitude < 0.01) {
        this.tempColor.setRGB(0.12, 0.14, 0.2);
      } else if (normalized >= 0) {
        const t = magnitude;
        this.tempColor.setRGB(
          lerp(0.38, 1.0, t),
          lerp(0.45, 0.95, t),
          lerp(0.25, 0.6, t),
        );
      } else {
        const t = magnitude;
        this.tempColor.setRGB(
          lerp(0.12, 0.38, t),
          lerp(0.2, 0.48, t),
          lerp(0.5, 1.0, t),
        );
      }
      group.mesh.setColorAt(index, this.tempColor);
    });
    group.mesh.instanceColor.needsUpdate = true;
  }

  handleResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  animate() {
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function softmax(values) {
  if (!values.length) return [];
  const maxVal = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - maxVal));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => (sum === 0 ? 0 : value / sum));
}

function maxAbsValue(values) {
  let max = 0;
  for (let i = 0; i < values.length; i += 1) {
    const magnitude = Math.abs(values[i]);
    if (magnitude > max) {
      max = magnitude;
    }
  }
  return max;
}
