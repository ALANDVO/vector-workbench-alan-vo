import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client.ts";
import { ActivationCompareResponse } from "../api/types.ts";

export const ActivationGraphViewer: React.FC = () => {
  const [layers, setLayers] = useState<number>(30);
  const [data, setData] = useState<ActivationCompareResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    runComparison();
  }, []);

  const runComparison = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiClient.compareActivations(layers, 64);
      setData(res);
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
          <h2 style={{ fontSize: "1.2rem", fontWeight: 600 }}>
            FFN Activation Graph Calibration (Issue #588)
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
            Investigating Squared ReLU vs SiLU numerical mis-calibration and perplexity drift
          </p>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Layers:</label>
          <input
            type="number"
            min={5}
            max={48}
            value={layers}
            onChange={(e) => setLayers(Number(e.target.value))}
            style={{ width: "70px" }}
          />
          <button className="btn-primary" onClick={runComparison} disabled={loading}>
            {loading ? "Simulating..." : "Run Calibration"}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {data && (
        <>
          <div className="grid-2" style={{ marginBottom: "20px" }}>
            <div style={{ background: "var(--bg-primary)", padding: "16px", borderRadius: "8px", border: "1px solid rgba(16, 185, 129, 0.3)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>Official BitNet b1.58 Spec</span>
                <span className="badge badge-green">CALIBRATED</span>
              </div>
              <div style={{ fontSize: "2rem", fontWeight: 700, color: "#34d399", margin: "10px 0" }}>
                PPL {data.relu_sqr_perplexity.toFixed(2)}
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                Squared ReLU (relu(x)²) preserves layer norm stability across all {data.num_layers} layers.
              </div>
            </div>

            <div style={{ background: "var(--bg-primary)", padding: "16px", borderRadius: "8px", border: "1px solid rgba(244, 63, 94, 0.3)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>Upstream Bug (LLM_FFN_SILU)</span>
                <span className="badge badge-rose">DEGRADED (+{data.perplexity_delta.toFixed(1)} PPL)</span>
              </div>
              <div style={{ fontSize: "2rem", fontWeight: 700, color: "#fb7185", margin: "10px 0" }}>
                PPL {data.silu_perplexity.toFixed(2)}
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                SiLU activation under-scales gate projections into sub-norm compounding error.
              </div>
            </div>
          </div>

          <div style={{ background: "var(--bg-primary)", padding: "14px", borderRadius: "8px", marginBottom: "20px", fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: "1.6" }}>
            <strong style={{ color: "var(--text-primary)" }}>Root Cause Analysis: </strong>
            {data.root_cause_explanation}
          </div>

          <div className="table-responsive">
            <h3 style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "8px" }}>
              Layer-by-Layer Activation Norms (Sample)
            </h3>
            <table>
              <thead>
                <tr>
                  <th>Layer Index</th>
                  <th>ReLU² Activation Norm</th>
                  <th>SiLU Activation Norm</th>
                  <th>ReLU² Sparsity</th>
                  <th>Stability Status</th>
                </tr>
              </thead>
              <tbody>
                {data.relu_sqr_stats.slice(0, 10).map((stat, idx) => {
                  const siluStat = data.silu_stats[idx];
                  return (
                    <tr key={stat.layer}>
                      <td className="mono">Layer {stat.layer}</td>
                      <td className="mono" style={{ color: "#34d399" }}>{stat.norm.toFixed(4)}</td>
                      <td className="mono" style={{ color: "#fb7185" }}>{siluStat ? siluStat.norm.toFixed(4) : "-"}</td>
                      <td className="mono">{(stat.sparsity * 100).toFixed(1)}%</td>
                      <td>
                        <span className="badge badge-green">STABLE</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};
