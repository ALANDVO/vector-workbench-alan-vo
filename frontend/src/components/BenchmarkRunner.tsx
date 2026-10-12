import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client.ts";
import { BenchmarkRun } from "../api/types.ts";

export const BenchmarkRunner: React.FC = () => {
  const [history, setHistory] = useState<BenchmarkRun[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState<"perplexity" | "quantization_fidelity" | "throughput">("quantization_fidelity");
  const [matrixDim, setMatrixDim] = useState<number>(128);
  const [loading, setLoading] = useState<boolean>(false);
  const [latestResult, setLatestResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const runs = await apiClient.getBenchmarkHistory();
      setHistory(runs);
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleRun = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiClient.runBenchmark(selectedBenchmark, matrixDim, 30);
      setLatestResult(res);
      await fetchHistory();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 600 }}>Ternary & Kernel Benchmark Suite</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
            Deterministic performance and compression evaluation across matrix kernels
          </p>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div style={{ display: "flex", gap: "16px", alignItems: "center", marginBottom: "20px" }}>
        <div style={{ flex: 1 }}>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
            Benchmark Target
          </label>
          <select
            value={selectedBenchmark}
            onChange={(e) => setSelectedBenchmark(e.target.value as any)}
            style={{ width: "100%" }}
          >
            <option value="quantization_fidelity">Quantization Fidelity & Memory Compression</option>
            <option value="throughput">Integer Addition Throughput (3-ary vs 5-ary LUT - Issue #569)</option>
            <option value="perplexity">Multi-Layer Transformer Perplexity</option>
          </select>
        </div>

        <div>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
            Matrix Dimension
          </label>
          <select
            value={matrixDim}
            onChange={(e) => setMatrixDim(Number(e.target.value))}
            style={{ width: "120px" }}
          >
            <option value={64}>64 x 64</option>
            <option value={128}>128 x 128</option>
            <option value={256}>256 x 256</option>
          </select>
        </div>

        <div style={{ alignSelf: "flex-end" }}>
          <button className="btn-primary" onClick={handleRun} disabled={loading}>
            {loading ? "Executing Run..." : "Execute Benchmark"}
          </button>
        </div>
      </div>

      {latestResult && (
        <div style={{ background: "var(--bg-primary)", padding: "16px", borderRadius: "8px", border: "1px solid var(--border-subtle)", marginBottom: "20px" }}>
          <h3 style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--accent-cyan)", marginBottom: "10px" }}>
            Latest Run Result ({latestResult.benchmark_type})
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "12px" }}>
            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>Metric</span>
              <div style={{ fontWeight: 600 }}>{latestResult.metric_name}</div>
            </div>
            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>Value</span>
              <div style={{ fontWeight: 700, color: "var(--accent-green)", fontSize: "1.1rem" }}>
                {latestResult.metric_value}
              </div>
            </div>
            {latestResult.details?.compression_ratio && (
              <div>
                <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>Compression Ratio</span>
                <div style={{ fontWeight: 600 }}>{latestResult.details.compression_ratio}x</div>
              </div>
            )}
            {latestResult.details?.speedup_5ary_vs_3ary && (
              <div>
                <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>5-Ary LUT Speedup</span>
                <div style={{ fontWeight: 600, color: "var(--accent-cyan)" }}>{latestResult.details.speedup_5ary_vs_3ary}x</div>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="table-responsive">
        <h3 style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "8px" }}>
          Persistent Experiment History
        </h3>
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Timestamp</th>
              <th>Suite</th>
              <th>Metric</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {history.map((run) => (
              <tr key={run.id}>
                <td className="mono">#{run.id}</td>
                <td className="mono" style={{ fontSize: "0.75rem" }}>{new Date(run.timestamp).toLocaleString()}</td>
                <td>
                  <span className="badge badge-blue">{run.benchmark_type}</span>
                </td>
                <td className="mono">{run.metric_name}</td>
                <td className="mono" style={{ color: "var(--accent-green)", fontWeight: 600 }}>
                  {run.metric_value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
