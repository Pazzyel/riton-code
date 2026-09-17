from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from directory import DB_PATH


CHECKPOINT_VERSION: int = 1


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MessageRecord:
    id: int
    session_id: str
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class CheckpointRecord:
    session_id: str
    agent_id: str
    agent_type: str
    version: int
    payload: Dict[str, Any]
    updated_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def one_line_message(content: str) -> str:
    return content.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


class PersistenceStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path: Path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS session (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS message (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES session(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS checkpoint (
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL CHECK (agent_type IN ('main', 'subagent')),
                    version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, agent_id),
                    FOREIGN KEY (session_id) REFERENCES session(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_message_session_order
                    ON message(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_session_updated
                    ON session(updated_at DESC);
                """
            )

    def create_session(self, session_id: str) -> None:
        timestamp: str = utc_now()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO session(session_id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, timestamp, timestamp),
            )

    def session_exists(self, session_id: str) -> bool:
        with closing(self._connect()) as connection, connection:
            row: Optional[sqlite3.Row] = connection.execute(
                "SELECT 1 FROM session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row is not None

    def list_sessions(self) -> List[SessionRecord]:
        with closing(self._connect()) as connection, connection:
            rows: List[sqlite3.Row] = connection.execute(
                "SELECT session_id, created_at, updated_at FROM session "
                "ORDER BY updated_at DESC, session_id ASC"
            ).fetchall()
        return [SessionRecord(**dict(row)) for row in rows]

    def list_messages(self, session_id: str) -> List[MessageRecord]:
        with closing(self._connect()) as connection, connection:
            rows: List[sqlite3.Row] = connection.execute(
                "SELECT id, session_id, role, content, created_at FROM message "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [MessageRecord(**dict(row)) for row in rows]


    def save_checkpoint(
        self,
        session_id: str,
        agent_id: str,
        agent_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        保存检查点

        主 agent 会在这些时机保存：
        1. (Optional) 会话初始化或恢复后
           新会话完成 SessionStart hook 后保存；旧会话加载状态后也立即保存一次。
        2. 用户消息进入状态时
           用户消息和主 checkpoint 在同一个 SQLite 事务中提交。
        3. 每次模型返回 assistant 消息后
           在执行 tool call 之前保存。这样如果工具执行期间崩溃，恢复时能发现未完成的调用。
        4. 每个工具调用形成终态结果后
           包括正常结果或被 hook 阻止。工具结果加入消息列表后立即保存。
        5. 每次 run_one_loop 结束后
           无论循环继续还是已生成最终回答，都会保存。
        6. (Optional)消息压缩后
           try_compact 改变消息历史后保存新的完整状态。
        7. 后台任务状态变化时
           后台任务创建和执行完成时都会触发主 checkpoint 保存，记录任务参数、状态及结果。
        8. (Optional) 恢复未完成工具调用时
           每个重放工具完成后保存，全部重放结束、更新轮次状态后再保存一次，固定修复的结果
        9. (Optional) Cron 消息注入后
           在 cron prompt 加入主消息历史后、进入 agent loop 前保存。
        10. (Optional) AI 最终回答写入可见历史时
            最终回答和 checkpoint 在同一个 SQLite 事务中提交。
        11. (Optional) 主程序正常退出输入循环时，取消 cron/queue 协程后再保存一次。

        子 agent 会在这些时机保存自己的 checkpoint：
        1. 创建或恢复 SubagentContext 后立即保存。
        2. (Optional) 消息压缩后。
        3. assistant 消息加入后、工具执行前。
        4. 每个工具完成或被阻止后。
        5. (Optional) 恢复并重放未完成工具后。
        6. 后台任务通知加入后。
        7. (Optional) continue、compact 或不可恢复错误等状态转换后。

        子 agent checkpoint 不保存后台任务；后台任务统一包含在主 agent checkpoint 中。
        """
        timestamp: str = utc_now()
        payload_text: str = json.dumps(payload, ensure_ascii=False)
        with closing(self._connect()) as connection, connection:
            self._upsert_checkpoint(
                connection,
                session_id,
                agent_id,
                agent_type,
                payload_text,
                timestamp,
            )

    def save_visible_message_and_checkpoint(
        self,
        session_id: str,
        role: str,
        content: str,
        agent_id: str,
        agent_type: str,
        payload: Dict[str, Any],
        only_if_last_user: bool = False,
    ) -> bool:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported visible message role: {role}")
        timestamp: str = utc_now()
        payload_text: str = json.dumps(payload, ensure_ascii=False)
        with closing(self._connect()) as connection, connection:
            if only_if_last_user:
                row: Optional[sqlite3.Row] = connection.execute(
                    "SELECT role FROM message WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                if row is None or row["role"] != "user":
                    self._upsert_checkpoint(
                        connection,
                        session_id,
                        agent_id,
                        agent_type,
                        payload_text,
                        timestamp,
                    )
                    return False
            connection.execute(
                "INSERT INTO message(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, timestamp),
            )
            connection.execute(
                "UPDATE session SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
            self._upsert_checkpoint(
                connection,
                session_id,
                agent_id,
                agent_type,
                payload_text,
                timestamp,
            )
        return True

    def save_parent_checkpoint_and_delete_child(
        self,
        session_id: str,
        parent_agent_id: str,
        parent_agent_type: str,
        parent_payload: Dict[str, Any],
        child_agent_id: str,
    ) -> None:
        """Atomically persist a resolved parent state and remove its child state."""
        timestamp: str = utc_now()
        payload_text: str = json.dumps(parent_payload, ensure_ascii=False)
        with closing(self._connect()) as connection, connection:
            self._upsert_checkpoint(
                connection,
                session_id,
                parent_agent_id,
                parent_agent_type,
                payload_text,
                timestamp,
            )
            connection.execute(
                "DELETE FROM checkpoint WHERE session_id = ? AND agent_id = ?",
                (session_id, child_agent_id),
            )

    def load_checkpoint(self, session_id: str, agent_id: str) -> Optional[CheckpointRecord]:
        with closing(self._connect()) as connection, connection:
            row: Optional[sqlite3.Row] = connection.execute(
                "SELECT session_id, agent_id, agent_type, version, payload, updated_at "
                "FROM checkpoint WHERE session_id = ? AND agent_id = ?",
                (session_id, agent_id),
            ).fetchone()
        if row is None:
            return None
        try:
            payload: Any = json.loads(row["payload"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Checkpoint JSON is invalid for agent {agent_id}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint payload is not an object for agent {agent_id}")
        version: int = int(row["version"])
        if version != CHECKPOINT_VERSION or payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version for agent {agent_id}: {version}"
            )
        return CheckpointRecord(
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            agent_type=row["agent_type"],
            version=version,
            payload=payload,
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _upsert_checkpoint(
        connection: sqlite3.Connection,
        session_id: str,
        agent_id: str,
        agent_type: str,
        payload_text: str,
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO checkpoint(session_id, agent_id, agent_type, version, payload, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, agent_id) DO UPDATE SET
                agent_type = excluded.agent_type,
                version = excluded.version,
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (
                session_id,
                agent_id,
                agent_type,
                CHECKPOINT_VERSION,
                payload_text,
                timestamp,
            ),
        )


_store: Optional[PersistenceStore] = None


def get_persistence_store() -> PersistenceStore:
    global _store
    if _store is None:
        _store = PersistenceStore()
    return _store


def set_persistence_store(store: Optional[PersistenceStore]) -> None:
    global _store
    _store = store
