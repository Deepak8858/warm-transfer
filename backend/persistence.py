"""Lightweight SQLite persistence for transfer summaries.

No external dependencies; uses stdlib sqlite3. Designed for low write volume.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid
import threading
import time
import logging
import os

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("PERSIST_DB_PATH", Path(__file__).parent / "data.db"))
_LOCK = threading.Lock()

DDL = """
CREATE TABLE IF NOT EXISTS transfer_summaries (
    id TEXT PRIMARY KEY,
    room_name TEXT NOT NULL,
    agent_a TEXT NOT NULL,
    agent_b TEXT,
    summary TEXT NOT NULL,
    call_context TEXT,
    created_at REAL NOT NULL
);
"""

def _connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(DDL)
        conn.commit()
    logger.info(f"SQLite persistence initialized at {DB_PATH}")

def create_transfer_record(room_name: str, agent_a: str, summary: str, call_context: str, agent_b: str | None = None) -> str:
    rec_id = str(uuid.uuid4())
    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO transfer_summaries (id, room_name, agent_a, agent_b, summary, call_context, created_at) VALUES (?,?,?,?,?,?,?)",
            (rec_id, room_name, agent_a, agent_b, summary, call_context, time.time())
        )
        conn.commit()
    return rec_id

def list_transfers(room_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    q = "SELECT id, room_name, agent_a, agent_b, summary, created_at FROM transfer_summaries"
    params: list[Any] = []
    if room_name:
        q += " WHERE room_name = ?"
        params.append(room_name)
    q += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _LOCK, _connect() as conn:
        rows = conn.execute(q, params).fetchall()
    keys = ["id","room_name","agent_a","agent_b","summary","created_at"]
    results = []
    for r in rows:
        d = dict(zip(keys, r))
        d['call_context'] = None
        results.append(d)
    return results

def get_transfer(rec_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT id, room_name, agent_a, agent_b, summary, call_context, created_at FROM transfer_summaries WHERE id = ?",
            (rec_id,)
        ).fetchone()
    if not row:
        return None
    keys = ["id","room_name","agent_a","agent_b","summary","call_context","created_at"]
    return dict(zip(keys, row))

def set_agent_b(rec_id: str, agent_b: str) -> bool:
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            "UPDATE transfer_summaries SET agent_b = ? WHERE id = ?",
            (agent_b, rec_id)
        )
        conn.commit()
        return cur.rowcount > 0
