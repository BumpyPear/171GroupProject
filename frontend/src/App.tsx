import './App.css'
import { useEffect, useState, type FormEvent } from 'react';
import { getFeatures, predict } from './api';


interface PredictionResult {
  predicted_quality?: number;
}


function App() {
  // for inputs
  type ModelKey = "polyreg" | "svm" | "rf" | "lgbm";

  // svm will be default model
  const [selectedModel, setSelectedModel] = useState<ModelKey>("svm");

  // list of feature names(technically could be hard coded)
  const [features, setFeatures] = useState<string[]>([]);
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);

  // in case of error
  const [error, setError] = useState<string | null>(null);


const models: { key: ModelKey; label: string }[] = [
  { key: "polyreg", label: "Polynomial Regression" },
  { key: "svm", label: "SVM Classification" },
  { key: "rf", label: "Random Forest" },
  { key: "lgbm", label: "Light GBM"},
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

    // build payload of input features
    const payload: Record<string, number> = {};
    for (const feat of features) {
      const raw = inputs[feat];
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

      {/*dropdown for models*/}
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

      {/*error fetching features*/}
      {error && <p style={{ color: "red" }}>{error}</p>}

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

    {/*display results*/}
    {result && (
      <div style={{ marginTop: "1.5rem", padding: "1rem", border: "1px solid #ccc" }}>
        {result.predicted_quality !== undefined && (
          <p>
            <strong>Predicted Quality: </strong>
            {result.predicted_quality.toFixed(2)}
          </p>
        )}
      </div>
    )}

    </div>
  );
}

export default App
