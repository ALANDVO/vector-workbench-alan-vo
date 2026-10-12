import React, { useState } from "react";
import { apiClient } from "../api/client.ts";
import { AdvisoryReportResponse } from "../api/types.ts";

export const LLMAdvisoryPanel: React.FC = () => {
  const [context, setContext] = useState<string>("Analyze the hardware memory savings and activation stability of BitNet b1.58.");
  const [report, setReport] = useState<AdvisoryReportResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiClient.generateAdvisory(context);
      setReport(res);
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
          <h2 style={{ fontSize: "1.2rem", fontWeight: 600 }}>Opt-In LLM Advisory Analysis</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
            Grounded architectural explanations (Offline deterministic core by default; optional LLM_API_KEY)
          </p>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div style={{ marginBottom: "16px" }}>
        <label style={{ display: "block", fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "4px" }}>
          Advisory Context Prompt
        </label>
        <textarea
          rows={2}
          value={context}
          onChange={(e) => setContext(e.target.value)}
          style={{ width: "100%", resize: "vertical" }}
        />
      </div>

      <div style={{ marginBottom: "20px" }}>
        <button className="btn-primary" onClick={handleGenerate} disabled={loading}>
          {loading ? "Generating Report..." : "Generate Advisory Report"}
        </button>
      </div>

      {report && (
        <div style={{ background: "var(--bg-primary)", padding: "18px", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <div>
              Provider: <span className="mono" style={{ color: "var(--accent-cyan)" }}>{report.provider}</span> ({report.model})
            </div>
            <span className="badge badge-amber">ADVISORY ONLY</span>
          </div>

          <div style={{ marginBottom: "14px", fontSize: "0.85rem" }}>
            <strong style={{ color: "var(--text-secondary)" }}>Grounded Benchmark Facts:</strong>
            <ul style={{ paddingLeft: "20px", marginTop: "6px", color: "var(--text-secondary)" }}>
              {report.grounded_findings.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          </div>

          <div style={{ color: "var(--text-primary)", fontSize: "0.9rem", lineHeight: "1.6", whiteSpace: "pre-wrap" }}>
            {report.analysis_text}
          </div>
        </div>
      )}
    </div>
  );
};
