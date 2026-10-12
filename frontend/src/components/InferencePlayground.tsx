import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client.ts";
import { CompletionResponse, ModelInfo } from "../api/types.ts";

export const InferencePlayground: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("bitnet-b1.58-2b-i2s");
  const [prompt, setPrompt] = useState<string>("Evaluate BitNet b1.58 inference throughput and memory scaling");
  const [activation, setActivation] = useState<"relu_sqr" | "silu">("relu_sqr");
  const [maxTokens, setMaxTokens] = useState<number>(64);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<CompletionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const data = await apiClient.getModels();
      setModels(data);
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handlePrewarm = async () => {
    try {
      setLoading(true);
      setError(null);
      await apiClient.loadModel(selectedModel);
      await loadModels();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    try {
      setLoading(true);
      setError(null);
      const res = await apiClient.createCompletion(selectedModel, prompt, maxTokens, activation);
      setResult(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const activeModelInfo = models.find((m) => m.id === selectedModel);

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 600 }}>In-Memory Persistent Inference Engine</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
            Zero-cold-start resident memory execution (Addresses Issue #206)
          </p>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <span className={`badge ${activeModelInfo?.is_loaded ? "badge-green" : "badge-amber"}`}>
            {activeModelInfo?.is_loaded ? "Resident in RAM" : "Cold (On-Disk)"}
          </span>
          <button className="btn-secondary" onClick={handlePrewarm} disabled={loading}>
            Pre-Warm Tensors
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "16px" }}>
        <div>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
            Active 1-Bit Model
          </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{ width: "100%" }}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.quant_type} - {m.memory_mb} MB)
              </option>
            ))}
          </select>
        </div>

        <div>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
            FFN Activation Graph
          </label>
          <select
            value={activation}
            onChange={(e) => setActivation(e.target.value as "relu_sqr" | "silu")}
            style={{ width: "100%" }}
          >
            <option value="relu_sqr">Squared ReLU (relu(x)² - Official Specification)</option>
            <option value="silu">SiLU (Graph Bug - Degraded Perplexity)</option>
          </select>
        </div>
      </div>

      <div style={{ marginBottom: "16px" }}>
        <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
          Inference Prompt
        </label>
        <textarea
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          style={{ width: "100%", resize: "vertical" }}
        />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Max Tokens:</label>
          <input
            type="number"
            min={8}
            max={512}
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            style={{ width: "90px" }}
          />
        </div>

        <button className="btn-primary" onClick={handleGenerate} disabled={loading}>
          {loading ? "Generating (TGEMM Accumulation)..." : "Run In-Memory Completion"}
        </button>
      </div>

      {result && (
        <div style={{ background: "var(--bg-primary)", padding: "16px", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "10px", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            <div>
              Generated Tokens: <strong style={{ color: "var(--text-primary)" }}>{result.usage.completion_tokens}</strong>
            </div>
            <div>
              Latency: <strong style={{ color: "var(--accent-cyan)" }}>{result.latency_ms} ms</strong>
            </div>
            <div>
              Throughput: <strong style={{ color: "var(--accent-green)" }}>{result.tokens_per_second} tok/s</strong>
            </div>
          </div>
          <div style={{ color: "var(--text-primary)", fontSize: "0.95rem", lineHeight: "1.6" }}>
            {result.choices[0]?.text}
          </div>
        </div>
      )}
    </div>
  );
};
