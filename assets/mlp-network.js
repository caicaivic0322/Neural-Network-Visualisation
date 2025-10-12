export class MLPNetwork {
  constructor(definition) {
    if (!definition.layers?.length) {
      throw new Error("Network definition must contain layers.");
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

  forward(pixels) {
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
