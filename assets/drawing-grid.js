import { clamp } from "./math-utils.js";

export class DrawingGrid {
  constructor(container, rows, cols, options = {}) {
    if (!container) {
      throw new Error("Grid container not found.");
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
      drawRadius: 1.25,
      eraseRadius: 1.25,
      drawStrength: 0.82,
      eraseStrength: 0.78,
      softness: 0.48,
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
    title.textContent = "Draw a Digit";
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
