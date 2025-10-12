export async function fetchNetworkConfig(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load network weights (${response.status})`);
  }
  return response.json();
}

export function displayError(message) {
  const chart = document.getElementById("predictionChart");
  if (chart) {
    chart.innerHTML = `<p class="error-text">${message}</p>`;
  }
}
