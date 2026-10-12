import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client.ts";
import { AuditLog } from "../api/types.ts";

export const AuditLogViewer: React.FC = () => {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [page, setPage] = useState<number>(1);
  const [actionFilter, setActionFilter] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchLogs();
  }, [page, actionFilter]);

  const fetchLogs = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiClient.getAuditLogs(page, 15, actionFilter || undefined);
      setLogs(res.data);
      setTotal(res.total);
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
          <h2 style={{ fontSize: "1.2rem", fontWeight: 600 }}>Security & Audit Trail</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
            Immutable transactional record with UTC timestamps and user attribution
          </p>
        </div>

        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <select value={actionFilter} onChange={(e) => setActionFilter(e.target.value)}>
            <option value="">All Actions</option>
            <option value="APP_STARTUP">APP_STARTUP</option>
            <option value="DEMO_LOGIN">DEMO_LOGIN</option>
            <option value="MODEL_LOAD">MODEL_LOAD</option>
            <option value="MODEL_UNLOAD">MODEL_UNLOAD</option>
          </select>
          <button className="btn-secondary" onClick={fetchLogs} disabled={loading}>
            Refresh
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="table-responsive">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Timestamp (UTC)</th>
              <th>Principal</th>
              <th>Action</th>
              <th>Resource</th>
              <th>Status</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log) => (
              <tr key={log.id}>
                <td className="mono">#{log.id}</td>
                <td className="mono" style={{ fontSize: "0.75rem" }}>{log.timestamp}</td>
                <td className="mono" style={{ color: "var(--accent-cyan)" }}>{log.user_id}</td>
                <td>
                  <span className="badge badge-blue">{log.action}</span>
                </td>
                <td className="mono" style={{ fontSize: "0.8rem" }}>{log.resource}</td>
                <td>
                  <span className={`badge ${log.status === "SUCCESS" ? "badge-green" : "badge-rose"}`}>
                    {log.status}
                  </span>
                </td>
                <td style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>{log.details}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "16px", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
        <div>Total recorded events: {total}</div>
        <div style={{ display: "flex", gap: "8px" }}>
          <button
            className="btn-secondary"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
          >
            Previous
          </button>
          <span style={{ alignSelf: "center" }}>Page {page}</span>
          <button
            className="btn-secondary"
            onClick={() => setPage((p) => p + 1)}
            disabled={page * 15 >= total}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
};
