import './App.css'
import { useEffect, useState, type FormEvent } from 'react';
import { getFeatures, predict } from './api';


interface PredictionResult {
  // for regression endpoints (polyreg & rf):
  predicted_quality?: number;
  // for classification endpoint (svm):
  predicted_class?: string;
  probability?: number;
}

function App() {
  // for inputs
  type ModelKey = "polyreg" | "svm" | "rf";

  // selectedModel defaults to the svm
  const [selectedModel, setSelectedModel] = useState<ModelKey>("svm");

  // 2) list of feature names (["fixed_acidity", …, "alcohol"])
  const [features, setFeatures] = useState<string[]>([]);

  // 3) inputs maps each feature → string (so we control the <input /> value)
  const [inputs, setInputs] = useState<Record<string, string>>({});

  // 4) store the prediction result JSON
  const [result, setResult] = useState<PredictionResult | null>(null);

  // 5) error message (e.g. missing_features, invalid_input, network error)
  const [error, setError] = useState<string | null>(null);


const models: { key: ModelKey; label: string }[] = [
  { key: "polyreg", label: "Polynomial Regression" },
  { key: "svm", label: "SVM Classification" },
  { key: "rf", label: "Random Forest" },
];


  useEffect(() => {
    setFeatures([]);
    setInputs({});
    setResult(null);
    setError(null);

    // technically this is not necessary because they all should have the same inputs
    getFeatures(selectedModel)
      .then((f: string[]) => {
        setFeatures(f);

        const initialInputs: Record<string, string> = {};
        f.forEach((feat: string) => {
          initialInputs[feat] = "";
        });
        setInputs(initialInputs)
      })
      .catch((err: unknown) => {
        const msg = 
          err instanceof Error
            ? err.message
            : "unknown error while fetching features";
        setError(msg);
      })  
  }, [selectedModel])

  const handleChange = (
    featKey: string,
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setInputs({
      ...inputs,
      [featKey]: e.target.value,
    });
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    // 1) Build payload: { feature: parseFloat(value), … }
    const payload: Record<string, number> = {};
    for (const feat of features) {
      const raw = inputs[feat];
      // parseFloat("") returns NaN, we let the backend catch invalid input
      payload[feat] = parseFloat(raw);
    }

    try {
      const resJson = await predict(selectedModel, payload);
      setResult(resJson as PredictionResult);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : "An unknown error occurred during prediction";
      setError(msg);
    }
  };


  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: "1rem" }}>
      <h1>Wine Quality Predictor</h1>

      {/* Model selector dropdown */}
      <label>
        Choose Model:{" "}
        <select
        className='dropdown'
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value as ModelKey)}
        >
          {models.map((m) => (
            <option key={m.key} value={m.key}>
              {m.label}
            </option>
          ))}
        </select>
      </label>

      <hr style={{ margin: "1rem 0" }} />

      {/* Show any error when fetching features */}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {/* If features loaded, show form; else show “Loading…” */}
      {features.length > 0 ? (
        <form onSubmit={handleSubmit}>
          {features.map((feat) => (
            <div
              key={feat}
              style={{
                display: "flex",
                alignItems: "center",
                marginBottom: "0.5rem",
              }}
            >
              <label style={{ flex: "1 0 150px" }}>{feat}:</label>
              <input
                className='input'
                type="number"
                step="any"
                required
                style={{ flex: "1 0 200px" }}
                value={inputs[feat]}
                onChange={(e) => handleChange(feat, e)}
              />
            </div>
          ))}

          <button type="submit" style={{ marginTop: "1rem" }}>
            Predict
          </button>
        </form>
      ) : (
        <p>Loading features...</p>
      )}

      {/* Display prediction result once available */}
      {result && (
        <div
          style={{
            marginTop: "1.5rem",
            padding: "1rem",
            border: "1px solid #ccc",
          }}
        >
          {(selectedModel === "polyreg" || selectedModel === "rf") && (
            <p>
              <strong>Predicted Quality: </strong>
              {result.predicted_quality}
            </p>
          )}

          {selectedModel === "svm" && result.predicted_class && (
            <p>
              <strong>Predicted Class: </strong>
              {result.predicted_class}
              {result.probability !== undefined && (
                <> - (Probability: {Number(result.probability).toFixed(2)})</>
              )}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default App
