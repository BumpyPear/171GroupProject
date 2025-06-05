// frontend/src/api.js

// Adjust BASE depending on DEV vs. PROD
const BASE = import.meta.env.DEV
  ? "http://localhost:5000/api"
  : "/api";

/**
 * Fetch the list of feature‐names that the given model expects.
 *
 * @param {string} model  One of "polyreg", "svm", or "rf".
 * @returns {Promise<string[]>}  e.g. ["fixed_acidity", "volatile_acidity", ..., "alcohol"]
 */
export async function getFeatures(model: string) {
  const res = await fetch(`${BASE}/features/${model}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch features for ${model}: ${res.statusText}`);
  }
  const { features } = await res.json();
  return features;
}

/**
 * Send a JSON payload of { feature1: value1, feature2: value2, ... }
 * to /api/predict/<model> and return the parsed JSON response.
 *
 * @param {string} model         "polyreg", "svm", or "rf"
 * @param {Object<string, number>} inputs  e.g. { fixed_acidity: 7.4, ... }
 * @returns {Promise<Object>}    e.g. { predicted_quality: 5.73 } or
 *                                { predicted_class: "High", probability: 0.87 }
 */
export async function predict(model: string, inputs: { [key:string] : number}) {
  const res = await fetch(`${BASE}/predict/${model}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(inputs),
  });
  if (!res.ok) {
    // If Flask returned a 400/500 with a JSON error, we parse it:
    let errJson;
    try {
      errJson = await res.json();
    } catch { /*  */ }
    throw new Error(errJson?.message || `Prediction failed (${res.status})`);
  }
  return res.json();
}
