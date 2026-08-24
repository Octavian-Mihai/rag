from __future__ import annotations

import sqlite3
from pathlib import Path

from rag.config import load_config
from rag.models import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    title TEXT,
    parties TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    page INTEGER NOT NULL,
    heading TEXT,
    text TEXT NOT NULL,
    is_parent INTEGER NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS clauses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    page INTEGER,
    heading TEXT,
    amount_usd REAL,
    term_months INTEGER,
    party TEXT,
    date_iso TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
"""


def db_path() -> Path:
    cfg = load_config()
    path = Path(cfg["sqlite"]["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert_document(doc_id: str, filename: str, title: str, parties: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (doc_id, filename, title, parties)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                filename=excluded.filename, title=excluded.title, parties=excluded.parties
            """,
            (doc_id, filename, title, parties),
        )


def replace_chunks(doc_id: str, parents: list[Chunk], children: list[Chunk]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        for chunk in parents + children:
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, doc_id, parent_id, page, heading, text, is_parent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.parent_id,
                    chunk.page,
                    chunk.heading,
                    chunk.text,
                    1 if chunk.is_parent else 0,
                ),
            )


def replace_clauses(doc_id: str, rows: list[dict]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM clauses WHERE doc_id = ?", (doc_id,))
        for row in rows:
            conn.execute(
                """
                INSERT INTO clauses (doc_id, page, heading, amount_usd, term_months, party, date_iso)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    row.get("page"),
                    row.get("heading"),
                    row.get("amount_usd"),
                    row.get("term_months"),
                    row.get("party"),
                    row.get("date_iso"),
                ),
            )


def get_parent(parent_id: str) -> Chunk | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ? AND is_parent = 1", (parent_id,)
        ).fetchone()
    if not row:
        return None
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        parent_id=row["parent_id"],
        page=row["page"],
        text=row["text"],
        heading=row["heading"],
        is_parent=True,
    )


def all_parent_chunks() -> list[Chunk]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM chunks WHERE is_parent = 1").fetchall()
    return [
        Chunk(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            parent_id=r["parent_id"],
            page=r["page"],
            text=r["text"],
            heading=r["heading"],
            is_parent=True,
        )
        for r in rows
    ]


def list_documents() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT doc_id, filename, title, parties FROM documents").fetchall()
    return [dict(r) for r in rows]
