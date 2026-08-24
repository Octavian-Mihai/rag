from __future__ import annotations

import re
from uuid import uuid4

from rag.models import Chunk

HEADING_RE = re.compile(
    r"^(article\s+\d+|section\s+\d+|clause\s+\d+|\d+\.\d+|[A-Z][A-Z0-9 \-]{8,})$",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    return bool(HEADING_RE.match(stripped))


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _pack_children(paragraphs: list[str], max_tokens: int) -> list[str]:
    children: list[str] = []
    buf: list[str] = []
    for para in paragraphs:
        candidate = "\n\n".join(buf + [para]) if buf else para
        if buf and estimate_tokens(candidate) > max_tokens:
            children.append("\n\n".join(buf))
            buf = [para]
        else:
            buf.append(para)
    if buf:
        children.append("\n\n".join(buf))
    return children


def semantic_parent_child(
    doc_id: str,
    pages: list[tuple[int, str]],
    child_max_tokens: int = 512,
) -> tuple[list[Chunk], list[Chunk]]:
    """Split pages into heading-bounded parents and packed child chunks."""
    parents: list[Chunk] = []
    children: list[Chunk] = []

    current_heading = "Preamble"
    current_page = pages[0][0] if pages else 1
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_page, current_lines
        text = "\n".join(current_lines).strip()
        if not text:
            return
        parent_id = str(uuid4())
        parent = Chunk(
            chunk_id=parent_id,
            doc_id=doc_id,
            parent_id=parent_id,
            page=current_page,
            text=text,
            heading=current_heading,
            is_parent=True,
        )
        parents.append(parent)
        for child_text in _pack_children(split_paragraphs(text), child_max_tokens):
            children.append(
                Chunk(
                    chunk_id=str(uuid4()),
                    doc_id=doc_id,
                    parent_id=parent_id,
                    page=current_page,
                    text=child_text,
                    heading=current_heading,
                    is_parent=False,
                )
            )

    for page_num, page_text in pages:
        for line in page_text.splitlines():
            if is_heading(line):
                flush()
                current_heading = line.strip()
                current_page = page_num
                current_lines = [line]
            else:
                if not current_lines:
                    current_page = page_num
                current_lines.append(line)
        current_lines.append("")

    flush()
    return parents, children


def fixed_size_chunks(doc_id: str, pages: list[tuple[int, str]], size: int = 400) -> list[Chunk]:
    words: list[tuple[int, str]] = []
    for page, text in pages:
        for word in text.split():
            words.append((page, word))
    out: list[Chunk] = []
    for i in range(0, len(words), size):
        window = words[i : i + size]
        cid = str(uuid4())
        out.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                parent_id=cid,
                page=window[0][0],
                text=" ".join(w for _, w in window),
                heading="fixed",
            )
        )
    return out


def sliding_window_chunks(
    doc_id: str, pages: list[tuple[int, str]], size: int = 400, stride: int = 200
) -> list[Chunk]:
    words: list[tuple[int, str]] = []
    for page, text in pages:
        for word in text.split():
            words.append((page, word))
    out: list[Chunk] = []
    i = 0
    while i < len(words):
        window = words[i : i + size]
        cid = str(uuid4())
        out.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                parent_id=cid,
                page=window[0][0],
                text=" ".join(w for _, w in window),
                heading="sliding",
            )
        )
        if i + size >= len(words):
            break
        i += stride
    return out
