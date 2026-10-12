import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client.ts";
import { CompilerDiagnosticResponse } from "../api/types.ts";

export const DiagnosticsMatrix: React.FC = () => {
  const [compiler, setCompiler] = useState<string>("clang");
  const [version, setVersion] = useState<string>("20.1.0");
  const [arch, setArch] = useState<"x86_64" | "aarch64">("x86_64");
  const [cflags, setCflags] = useState<string>("-O3, -Wunreachable-code-break");
  const [result, setResult] = useState<CompilerDiagnosticResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    runEvaluation();
  }, []);

  const runEvaluation = async () => {
    try {
      setLoading(true);
      setError(null);
      const flagList = cflags.split(",").map((s) => s.trim()).filter(Boolean);
      const res = await apiClient.evaluateCompiler(compiler, version, arch, flagList);
      setResult(res);
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
          <h2 style={{ fontSize: "1.2rem", fontWeight: 600 }}>Compiler & Build Matrix Diagnostics</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
            Automated detection and resolution for compiler traps (Issues #180, #185, #334, #480)
          </p>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "16px", marginBottom: "16px" }}>
        <div>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
            Compiler
          </label>
          <select value={compiler} onChange={(e) => setCompiler(e.target.value)} style={{ width: "100%" }}>
            <option value="clang">Clang (LLVM)</option>
            <option value="gcc">GCC</option>
          </select>
        </div>

        <div>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
            Version
          </label>
          <input
            type="text"
            value={version}
            onChange={(e) => setVersion(e.target.value)}
            style={{ width: "100%" }}
            placeholder="e.g. 20.1.0 or 18.1.8"
          />
        </div>

        <div>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
            Target Architecture
          </label>
          <select value={arch} onChange={(e) => setArch(e.target.value as any)} style={{ width: "100%" }}>
            <option value="x86_64">x86_64 (AVX2 / AVX-512)</option>
            <option value="aarch64">aarch64 (ARM NEON / TL1)</option>
          </select>
        </div>

        <div>
          <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
            Compiler Flags (comma-separated)
          </label>
          <input
            type="text"
            value={cflags}
            onChange={(e) => setCflags(e.target.value)}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <button className="btn-primary" onClick={runEvaluation} disabled={loading}>
          {loading ? "Analyzing Toolchain..." : "Run Matrix Diagnostics"}
        </button>
      </div>

      {result && (
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
            <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>Toolchain Safety Assessment:</span>
            <span className={`badge ${result.safe_to_compile ? "badge-green" : "badge-rose"}`}>
              {result.safe_to_compile ? "SAFE TO COMPILE" : "BUILD TRAPS DETECTED"}
            </span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginBottom: "20px" }}>
            {result.hazards.map((h) => (
              <div
                key={h.issue_code}
                style={{
                  background: "var(--bg-primary)",
                  padding: "14px 18px",
                  borderRadius: "8px",
                  borderLeft: `4px solid ${h.severity === "error" ? "var(--accent-rose)" : "var(--accent-amber)"}`,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                  <strong style={{ fontSize: "0.95rem", color: "var(--text-primary)" }}>{h.title}</strong>
                  <span className={`badge ${h.severity === "error" ? "badge-rose" : "badge-amber"}`}>
                    {h.issue_code}
                  </span>
                </div>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "8px" }}>
                  {h.description}
                </p>
                <div style={{ fontSize: "0.8rem", color: "var(--accent-cyan)", background: "rgba(6, 182, 212, 0.05)", padding: "8px", borderRadius: "4px" }}>
                  <strong>Fix: </strong> {h.recommended_fix}
                </div>
              </div>
            ))}
          </div>

          <div style={{ background: "var(--bg-primary)", padding: "16px", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
            <h3 style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "8px" }}>
              Recommended CMake Presets
            </h3>
            <pre className="mono" style={{ fontSize: "0.8rem", color: "var(--accent-green)", overflowX: "auto" }}>
              {Object.entries(result.recommended_cmake_flags)
                .map(([k, v]) => `-D${k}=${v}`)
                .join(" \\\n  ")}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
