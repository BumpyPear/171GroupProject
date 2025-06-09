const BASE = import.meta.env.DEV
  ? "http://localhost:5000/api"
  : "/api";

  // get list of features
export async function getFeatures(model: string) {
  const res = await fetch(`${BASE}/features/${model}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch features for ${model}: ${res.statusText}`);
  }
  const { features } = await res.json();
  return features;
}

// send payload of inputs and get back the result
export async function predict(model: string, inputs: { [key:string] : number}) {
  const res = await fetch(`${BASE}/predict/${model}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(inputs),
  });
  if (!res.ok) {
    // for if the flask app returns an error
    let errJson;
    try {
      errJson = await res.json();
    } catch { /*  */ }
    throw new Error(errJson?.message || `Prediction failed (${res.status})`);
  }
  return res.json();
}
