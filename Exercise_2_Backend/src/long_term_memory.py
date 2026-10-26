"""SQLite-backed anonymous session memory and privacy-safe preferences."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock


class SQLiteConversationMemory:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS preferences (
                    session_id TEXT PRIMARY KEY,
                    response_detail TEXT NOT NULL DEFAULT 'standard',
                    experience_level TEXT NOT NULL DEFAULT 'beginner',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
                """
            )
            self._connection.commit()

    @staticmethod
    def new_session_id() -> str:
        return f"session-{uuid.uuid4().hex}"

    def ensure_session(self, session_id: str | None = None) -> str:
        session_id = session_id or self.new_session_id()
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO sessions(session_id, created_at) VALUES (?, ?)",
                (session_id, created_at),
            )
            self._connection.commit()
        return session_id

    def save_message(self, session_id: str, role: str, message: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Invalid message role")
        with self._lock:
            self.ensure_session(session_id)
            self._connection.execute(
                "INSERT INTO messages(session_id, role, message, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, str(message), datetime.now(timezone.utc).isoformat()),
            )
            self._connection.commit()

    def load_recent_messages(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT role, message, created_at FROM messages
                   WHERE session_id = ? ORDER BY message_id DESC LIMIT ?""",
                (session_id, int(limit)),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def save_session_summary(self, session_id: str, summary: str) -> None:
        with self._lock:
            self.ensure_session(session_id)
            self._connection.execute(
                """INSERT INTO session_summaries(session_id, summary, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                   summary=excluded.summary, updated_at=excluded.updated_at""",
                (session_id, str(summary), datetime.now(timezone.utc).isoformat()),
            )
            self._connection.commit()

    def load_session_summary(self, session_id: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT summary FROM session_summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
        return None if row is None else str(row["summary"])

    def save_preferences(
        self,
        session_id: str,
        response_detail: str = "standard",
        experience_level: str = "beginner",
    ) -> None:
        if response_detail not in {"short", "standard", "detailed"}:
            raise ValueError("Unsupported response_detail")
        if experience_level not in {"beginner", "intermediate"}:
            raise ValueError("Unsupported experience_level")
        with self._lock:
            self.ensure_session(session_id)
            self._connection.execute(
                """INSERT INTO preferences(session_id, response_detail, experience_level, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                   response_detail=excluded.response_detail,
                   experience_level=excluded.experience_level,
                   updated_at=excluded.updated_at""",
                (
                    session_id,
                    response_detail,
                    experience_level,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._connection.commit()

    def load_preferences(self, session_id: str) -> dict[str, str]:
        with self._lock:
            row = self._connection.execute(
                """SELECT response_detail, experience_level FROM preferences
                   WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
        return (
            {"response_detail": "standard", "experience_level": "beginner"}
            if row is None
            else dict(row)
        )

    def delete_session(self, session_id: str) -> None:
        """Permanently delete all persisted records for one anonymous session."""
        with self._lock:
            try:
                self._connection.execute("BEGIN")
                self._connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                self._connection.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
                self._connection.execute("DELETE FROM preferences WHERE session_id = ?", (session_id,))
                self._connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def export_schema_summary(self) -> str:
        with self._lock:
            tables = self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return json.dumps([row["name"] for row in tables])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteConversationMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
