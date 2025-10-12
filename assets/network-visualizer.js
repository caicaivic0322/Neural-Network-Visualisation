import { clamp, lerp, maxAbsValue } from "./math-utils.js";

const DEFAULT_LAYOUT = {
  layerSpacing: 5.8,
  inputSpacing: 0.26,
  hiddenSpacing: 1.0,
  inputNodeSize: 0.19,
  hiddenNodeRadius: 0.24,
  maxConnectionsPerNeuron: 20,
  connectionRadius: 0.018,
};

export class NetworkVisualizer {
  constructor(mlp, layoutOverrides) {
    this.mlp = mlp;
    this.layout = Object.assign({}, DEFAULT_LAYOUT, layoutOverrides || {});
    this.layerMeshes = [];
    this.connectionGroups = [];
    this.workObject3D = new THREE.Object3D();
    this.workColor = new THREE.Color();
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
    this.camera.position.set(14, 10, 24);

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 52;
    this.controls.target.set(0, 6, 0);

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
      this.layout.inputNodeSize,
      this.layout.inputNodeSize,
      this.layout.inputNodeSize,
    );
    const hiddenGeometry = new THREE.SphereGeometry(this.layout.hiddenNodeRadius, 16, 16);
    const baseMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
    baseMaterial.vertexColors = true;

    const layerCount = this.mlp.architecture.length;
    const totalWidth = (layerCount - 1) * this.layout.layerSpacing;
    const startX = -totalWidth / 2;

    this.mlp.architecture.forEach((neuronCount, layerIndex) => {
      const geometry = layerIndex === 0 ? inputGeometry : hiddenGeometry;
      const material = baseMaterial.clone();
      const mesh = new THREE.InstancedMesh(geometry, material, neuronCount);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(neuronCount * 3), 3);
      colorAttribute.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = colorAttribute;
      mesh.geometry.setAttribute("instanceColor", colorAttribute);

      const layerX = startX + layerIndex * this.layout.layerSpacing;
      const positions = this.computeLayerPositions(layerIndex, neuronCount, layerX);

      positions.forEach((position, instanceIndex) => {
        this.workObject3D.position.copy(position);
        this.workObject3D.updateMatrix();
        mesh.setMatrixAt(instanceIndex, this.workObject3D.matrix);
        const baseColor = this.workColor.setRGB(1, 1, 1);
        mesh.setColorAt(instanceIndex, baseColor);
      });

      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor.needsUpdate = true;
      this.scene.add(mesh);
      this.layerMeshes.push({ mesh, positions });
    });
  }

  computeLayerPositions(layerIndex, neuronCount, layerX) {
    const positions = [];
    if (layerIndex === 0) {
      const spacing = this.layout.inputSpacing;
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
      const spacing = this.layout.hiddenSpacing;
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
    const connectionRadius = this.layout.connectionRadius ?? 0.02;
    const geometry = new THREE.CylinderGeometry(connectionRadius, connectionRadius, 1, 10, 1, true);
    const material = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
      vertexColors: true,
    });

    this.mlp.layers.forEach((layer, layerIndex) => {
      const { selected, maxAbsWeight } = this.findImportantConnections(layer);
      if (!selected.length) return;

      const mesh = new THREE.InstancedMesh(geometry, material.clone(), selected.length);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      const colorAttribute = new THREE.InstancedBufferAttribute(new Float32Array(selected.length * 3), 3);
      colorAttribute.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = colorAttribute;
      mesh.geometry.setAttribute("instanceColor", colorAttribute);

      selected.forEach((connection, instanceIndex) => {
        const sourcePosition = this.layerMeshes[layerIndex].positions[connection.sourceIndex];
        const targetPosition = this.layerMeshes[layerIndex + 1].positions[connection.targetIndex];
        const direction = targetPosition.clone().sub(sourcePosition);
        const length = direction.length();
        const midpoint = sourcePosition.clone().addScaledVector(direction, 0.5);

        this.workObject3D.position.copy(midpoint);
        const quaternion = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          direction.clone().normalize(),
        );
        this.workObject3D.scale.set(1, length, 1);
        this.workObject3D.quaternion.copy(quaternion);
        this.workObject3D.updateMatrix();
        mesh.setMatrixAt(instanceIndex, this.workObject3D.matrix);
        mesh.setColorAt(instanceIndex, this.workColor.setRGB(1, 1, 1));
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
    const limit = this.layout.maxConnectionsPerNeuron;
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

  update(activations) {
    this.layerMeshes.forEach((layer, layerIndex) => {
      const values = activations[layerIndex];
      if (!values) return;
      const scale = layerIndex === 0 ? 2.5 : maxAbsValue(activations[layerIndex]);
      this.applyNodeColors(layer.mesh, values, scale || 1);
    });

    this.connectionGroups.forEach((group) => {
      const sourceValues = activations[group.sourceLayer];
      if (!sourceValues) return;
      this.applyConnectionColors(group, sourceValues);
    });
  }

  applyNodeColors(mesh, values, scale) {
    const safeScale = scale > 1e-6 ? scale : 1;
    for (let i = 0; i < values.length; i += 1) {
      const value = values[i];
      const normalized = clamp(value / safeScale, -1, 1);
      const magnitude = Math.abs(normalized);
      if (magnitude < 0.02) {
        this.workColor.setRGB(0.12, 0.14, 0.2);
      } else if (normalized >= 0) {
        const t = magnitude;
        this.workColor.setRGB(
          lerp(0.35, 1.0, t),
          lerp(0.4, 0.98, t),
          lerp(0.28, 0.65, t),
        );
      } else {
        const t = magnitude;
        this.workColor.setRGB(
          lerp(0.1, 0.35, t),
          lerp(0.18, 0.45, t),
          lerp(0.45, 1.0, t),
        );
      }
      mesh.setColorAt(i, this.workColor);
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
        this.workColor.setRGB(0.12, 0.14, 0.2);
      } else if (normalized >= 0) {
        const t = magnitude;
        this.workColor.setRGB(
          lerp(0.38, 1.0, t),
          lerp(0.45, 0.95, t),
          lerp(0.25, 0.6, t),
        );
      } else {
        const t = magnitude;
        this.workColor.setRGB(
          lerp(0.12, 0.38, t),
          lerp(0.2, 0.48, t),
          lerp(0.5, 1.0, t),
        );
      }
      group.mesh.setColorAt(index, this.workColor);
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
