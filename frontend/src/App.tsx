import React, { useEffect, useState } from "react";
import { apiClient } from "./api/client.ts";
import { UserProfile } from "./api/types.ts";
import { ActivationGraphViewer } from "./components/ActivationGraphViewer.tsx";
import { AuditLogViewer } from "./components/AuditLogViewer.tsx";
import { BenchmarkRunner } from "./components/BenchmarkRunner.tsx";
import { DiagnosticsMatrix } from "./components/DiagnosticsMatrix.tsx";
import { Header } from "./components/Header.tsx";
import { InferencePlayground } from "./components/InferencePlayground.tsx";
import { LLMAdvisoryPanel } from "./components/LLMAdvisoryPanel.tsx";
import { LUTBenchmarkProfiler } from "./components/LUTBenchmarkProfiler.tsx";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<
    "inference" | "activation" | "benchmarks" | "lut_benchmarks" | "diagnostics" | "audit" | "advisory"
  >("inference");
  const [user, setUser] = useState<UserProfile | null>(null);
  const [activeRole, setActiveRole] = useState<"viewer" | "operator" | "admin">("operator");
  const [initLoading, setInitLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    initAuth(activeRole);
  }, []);

  const initAuth = async (role: "viewer" | "operator" | "admin") => {
    try {
      setInitLoading(true);
      setError(null);
      await apiClient.loginWithDemo(role);
      const profile = await apiClient.getProfile();
      setUser(profile);
      setActiveRole(role);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setInitLoading(false);
    }
  };

  const handleRoleChange = async (role: "viewer" | "operator" | "admin") => {
    await initAuth(role);
  };

  return (
    <div className="container">
      <Header user={user} activeRole={activeRole} onRoleChange={handleRoleChange} />

      {error && <div className="error-banner">Session Error: {error}</div>}

      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === "inference" ? "active" : ""}`}
          onClick={() => setActiveTab("inference")}
        >
          Inference Engine (Issue #206)
        </button>
        <button
          className={`tab-btn ${activeTab === "activation" ? "active" : ""}`}
          onClick={() => setActiveTab("activation")}
        >
          Activation Calibration (Issue #588)
        </button>
        <button
          className={`tab-btn ${activeTab === "benchmarks" ? "active" : ""}`}
          onClick={() => setActiveTab("benchmarks")}
        >
        </button>
        <button
          className={`tab-btn ${activeTab === "lut_benchmarks" ? "active" : ""}`}
          onClick={() => setActiveTab("lut_benchmarks")}
        >
          LUT Kernel Profiler (Issue #569)
        </button>
        <button
          className={`tab-btn ${activeTab === "diagnostics" ? "active" : ""}`}
          onClick={() => setActiveTab("diagnostics")}
        >
          Compiler Diagnostics (Issues #180, #480)
        </button>
        <button
          className={`tab-btn ${activeTab === "audit" ? "active" : ""}`}
          onClick={() => setActiveTab("audit")}
        >
          Audit Trail
        </button>
        <button
          className={`tab-btn ${activeTab === "advisory" ? "active" : ""}`}
          onClick={() => setActiveTab("advisory")}
        >
          LLM Advisory
        </button>
      </nav>

      <main>
        {initLoading ? (
          <div className="card" style={{ textAlign: "center", padding: "40px" }}>
            <div style={{ color: "var(--accent-cyan)", fontWeight: 600 }}>Initializing In-Memory Session...</div>
          </div>
        ) : (
          <>
            {activeTab === "inference" && <InferencePlayground />}
            {activeTab === "activation" && <ActivationGraphViewer />}
            {activeTab === "benchmarks" && <BenchmarkRunner />}
            {activeTab === "lut_benchmarks" && <LUTBenchmarkProfiler />}
            {activeTab === "diagnostics" && <DiagnosticsMatrix />}
            {activeTab === "audit" && <AuditLogViewer />}
            {activeTab === "advisory" && <LLMAdvisoryPanel />}
          </>
        )}
      </main>

      <footer style={{ marginTop: "40px", textAlign: "center", fontSize: "0.8rem", color: "var(--text-muted)" }}>
        Vector Workbench — Built by <a href="mailto:alanvo@gmail.com" style={{ color: "var(--accent-cyan)", textDecoration: "none" }}>Alan Vo</a> &lt;alanvo@gmail.com&gt;
      </footer>
    </div>
  );
};

export default App;
