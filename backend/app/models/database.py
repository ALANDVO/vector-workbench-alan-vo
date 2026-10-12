"""SQLite persistent storage, migrations, and audit logging."""

import datetime as dt
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from app.core.config import settings


def get_db_path() -> str:
    """Extract SQLite database file path from URL."""
    url = settings.database_url
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "")
        if path == ":memory:":
            return path
        return os.path.abspath(path)
    return "vector_workbench.sqlite"


class Database:
    """Parameterized SQLite manager with audit trails and transactional execution."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_db_path()
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_tables(self) -> None:
        """Create database tables if they do not exist."""
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    parameter_count INTEGER NOT NULL,
                    quant_type TEXT NOT NULL,
                    memory_mb REAL NOT NULL,
                    is_loaded INTEGER NOT NULL DEFAULT 0,
                    loaded_at TEXT
                );

                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    benchmark_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    details_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inference_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    completion TEXT NOT NULL,
                    tokens_generated INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    user_id TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def log_audit(
        self,
        user_id: str,
        action: str,
        resource: str,
        status: str,
        details: str = "",
    ) -> int:
        """Record an audit trail event with UTC timestamp."""
        now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_logs (timestamp, user_id, action, resource, status, details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (now_utc, user_id, action, resource, status, details),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_audit_logs(
        self,
        page: int = 1,
        page_size: int = 20,
        action: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Query audit logs with filtering and pagination."""
        offset = (page - 1) * page_size
        query = "SELECT * FROM audit_logs WHERE 1=1"
        count_query = "SELECT COUNT(*) FROM audit_logs WHERE 1=1"
        params: List[Any] = []

        if action:
            query += " AND action = ?"
            count_query += " AND action = ?"
            params.append(action)

        if status_filter:
            query += " AND status = ?"
            count_query += " AND status = ?"
            params.append(status_filter)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]

            cursor.execute(query, params + [page_size, offset])
            rows = [dict(row) for row in cursor.fetchall()]

            return rows, total

    def record_benchmark_run(
        self,
        benchmark_type: str,
        metric_name: str,
        metric_value: float,
        details: Dict[str, Any],
    ) -> int:
        """Persist a completed benchmark experiment record."""
        now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO benchmark_runs (timestamp, benchmark_type, metric_name, metric_value, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    now_utc,
                    benchmark_type,
                    metric_name,
                    metric_value,
                    json.dumps(details),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_benchmark_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent benchmark runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["details"] = json.loads(d["details_json"])
                results.append(d)
            return results

    def record_inference_session(
        self,
        model_id: str,
        prompt: str,
        completion: str,
        tokens: int,
        latency_ms: float,
        user_id: str,
    ) -> int:
        """Record an in-memory inference generation event."""
        now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO inference_sessions (timestamp, model_id, prompt, completion, tokens_generated, latency_ms, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_utc,
                    model_id,
                    prompt,
                    completion,
                    tokens,
                    latency_ms,
                    user_id,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0


db = Database()
