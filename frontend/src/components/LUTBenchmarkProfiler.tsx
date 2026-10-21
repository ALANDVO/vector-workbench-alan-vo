import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client.ts";
import { LUTBenchmarkResponse } from "../api/types.ts";

const D = { m: "Activation Rows", n: "Shared Dim", k: "Weight Cols" };

export const LUTBenchmarkProfiler: React.FC = () => {
  const [d, setD] = useState({ m: 128, n: 128, k: 128 });
  const [r, setR] = useState<LUTBenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    run();
  }, []);

  const run = async () => {
    try {
      setLoading(true);
      setError(null);
      setR(await apiClient.runLUTKernelBenchmark(d.m, d.n, d.k));
    } catch (x: unknown) {
      setError(x instanceof Error ? x.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const f = (n: number) =>
    n >= 1e6
      ? (n / 1e6).toFixed(2) + "M"
      : n >= 1e3
      ? (n / 1e3).toFixed(1) + "K"
      : n.toFixed(0);

  const lt = (u: number) =>
    u < 1e3
      ? `${u.toFixed(1)}µs`
      : u < 1e6
      ? `${(u / 1e3).toFixed(1)}ms`
      : `${(u / 1e6).toFixed(2)}s`;

  const c = (x: number) =>
    x >= 1.5
      ? "var(--accent-green)"
      : x >= 1.2
      ? "var(--accent-cyan)"
      : x >= 1
      ? "var(--accent-amber)"
      : "var(--accent-rose)";

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 600 }}>LUT Kernel Profiler (Issue #569)</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: ".85rem" }}>
            AVX-512 5-ary vs 3-ary vs scalar ternary GEMM
          </p>
        </div>
      </div>
      {error && <div className="error-banner">{error}</div>}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 16, marginBottom: 16 }}>
        {["m", "n", "k"].map((k) => (
          <div key={k}>
            <label style={{ display: "block", fontSize: ".85rem", color: "var(--text-secondary)", marginBottom: 4 }}>
              Matrix {k.toUpperCase()} ({D[k as keyof typeof D]})
            </label>
            <input
              type="number"
              value={d[k as keyof typeof d]}
              onChange={(e) => {
                const v = Math.max(16, Math.min(512, parseInt(e.target.value) || 0));
                setD((o) => ({ ...o, [k]: v }));
              }}
              style={{ width: "100%" }}
              min={16}
              max={512}
            />
          </div>
        ))}
      </div>
      <div style={{ marginBottom: 20 }}>
        <button className="btn-primary" onClick={run} disabled={loading}>
          {loading ? "Profiling..." : "Run LUT Benchmark"}
        </button>
      </div>
      {r && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))", gap: 16, marginBottom: 24 }}>
            {["3-ary vs Scalar", "5-ary vs Scalar", "5-ary vs 3-ary"].map((l, i) => {
              const v = i == 0 ? r.speedup_3ary_vs_scalar : i == 1 ? r.speedup_5ary_vs_scalar : r.speedup_5ary_vs_3ary;
              return (
                <div
                  key={l}
                  style={{
                    background: "var(--bg-primary)",
                    padding: 16,
                    borderRadius: 8,
                    border: "1px solid var(--border-subtle)",
                  }}
                >
                  <div style={{ fontSize: ".8rem", color: "var(--text-secondary)", marginBottom: 4 }}>{l}</div>
                  <div style={{ fontSize: "2rem", fontWeight: 700, color: c(v) }}>{v.toFixed(2)}x</div>
                </div>
              );
            })}
          </div>
          <div style={{ marginBottom: 24 }}>
            <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 12, color: "var(--text-primary)" }}>
              Kernel Execution Details
            </h3>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: ".85rem" }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid var(--border-subtle)" }}>
                    <th style={{ textAlign: "left", padding: "10px 12px", color: "var(--text-secondary)" }}>Kernel</th>
                    <th style={{ textAlign: "right", padding: "10px 12px", color: "var(--text-secondary)" }}>Latency</th>
                    <th style={{ textAlign: "right", padding: "10px 12px", color: "var(--text-secondary)" }}>Additions</th>
                    <th style={{ textAlign: "right", padding: "10px 12px", color: "var(--text-secondary)" }}>LUT Lookups</th>
                    <th style={{ textAlign: "right", padding: "10px 12px", color: "var(--text-secondary)" }}>Bandwidth (MB/s)</th>
                    <th style={{ textAlign: "right", padding: "10px 12px", color: "var(--text-secondary)" }}>Op. Intensity</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "10px 12px", fontWeight: 600 }}>Scalar</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{lt(r.scalar_result.latency_microseconds)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{f(r.scalar_result.additions_count)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{f(r.scalar_result.lut_lookups)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{r.bandwidth_scalar_mb_s.toFixed(1)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{r.operational_intensity_scalar.toFixed(2)}</td>
                  </tr>
                  <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "10px 12px", fontWeight: 600 }}>3-ary LUT</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{lt(r.lut3_result.latency_microseconds)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{f(r.lut3_result.additions_count)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{f(r.lut3_result.lut_lookups)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{r.bandwidth_lut3_mb_s.toFixed(1)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{r.operational_intensity_lut3.toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td style={{ padding: "10px 12px", fontWeight: 600 }}>5-ary LUT</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{lt(r.lut5_result.latency_microseconds)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{f(r.lut5_result.additions_count)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{f(r.lut5_result.lut_lookups)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{r.bandwidth_lut5_mb_s.toFixed(1)}</td>
                    <td style={{ textAlign: "right", padding: "10px 12px" }}>{r.operational_intensity_lut5.toFixed(2)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};