import React from "react";
import { UserProfile } from "../api/types.ts";

interface HeaderProps {
  user: UserProfile | null;
  activeRole: "viewer" | "operator" | "admin";
  onRoleChange: (role: "viewer" | "operator" | "admin") => void;
}

export const Header: React.FC<HeaderProps> = ({ user, activeRole, onRoleChange }) => {
  return (
    <header className="header-banner">
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <h1 style={{ fontSize: "1.4rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
            Vector Workbench
          </h1>
          <span className="badge badge-blue">1-Bit & Ternary LLM Engine</span>
        </div>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginTop: "4px" }}>
          Inference Runtime Server, FFN Squared ReLU Graph Validator & Compiler Diagnostics
        </p>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <div style={{ textAlign: "right", fontSize: "0.8rem" }}>
          <div style={{ color: "var(--text-secondary)" }}>
            Session: <span className="mono" style={{ color: "var(--text-primary)" }}>{user?.username || "demo-session"}</span>
          </div>
          <div style={{ marginTop: "2px" }}>
            Role: <span className="badge badge-green">{activeRole.toUpperCase()}</span>
          </div>
        </div>

        <div style={{ display: "flex", gap: "6px" }}>
          {(["viewer", "operator", "admin"] as const).map((r) => (
            <button
              key={r}
              className={`btn-secondary ${activeRole === r ? "btn-primary" : ""}`}
              style={{ padding: "6px 10px", fontSize: "0.75rem" }}
              onClick={() => onRoleChange(r)}
            >
              {r}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
};
