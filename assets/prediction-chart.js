export class PredictionChart {
  constructor(container) {
    this.container = container;
    this.rows = [];
    if (!this.container) {
      throw new Error("Prediction chart container not found.");
    }
    this.build();
  }

  build() {
    this.container.innerHTML = "";
    const title = document.createElement("h3");
    title.textContent = "Digit Probabilities";
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
