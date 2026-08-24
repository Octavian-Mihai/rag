from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from rag.chunking import semantic_parent_child
from rag.config import load_config
from rag.embedder import embed_texts
from rag.extract import extract_clause_rows, extract_parties, title_from_filename
from rag.graph import GRAPH
from rag.sparse import BM25
from rag.store import replace_chunks, replace_clauses, upsert_document
from rag.vectorstore import upsert_chunks


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def ingest_pdf(path: Path, doc_id: str | None = None) -> dict:
    cfg = load_config()
    path = Path(path)
    doc_id = doc_id or (path.stem.strip().replace(" ", "_") or str(uuid4()))
    pages = read_pdf_pages(path)
    if not any(t.strip() for _, t in pages):
        raise ValueError(f"No text extracted from {path}")

    child_max = int(cfg["chunking"]["child_max_tokens"])
    parents, children = semantic_parent_child(doc_id, pages, child_max_tokens=child_max)
    full_text = "\n".join(t for _, t in pages)
    parties = extract_parties(full_text)
    upsert_document(doc_id, path.name, title_from_filename(path), "; ".join(parties))
    replace_chunks(doc_id, parents, children)
    replace_clauses(doc_id, extract_clause_rows(parents))

    for child in children:
        BM25.fit_document(child.text)
    BM25.save()
    dense = embed_texts([c.text for c in children]).tolist()
    sparse_pairs = [BM25.encode(c.text) for c in children]
    upsert_chunks(children, dense, sparse_pairs)
    GRAPH.rebuild()
    return {
        "doc_id": doc_id,
        "filename": path.name,
        "pages": len(pages),
        "parents": len(parents),
        "children": len(children),
        "parties": parties,
    }


def ingest_directory(folder: Path) -> list[dict]:
    results = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        results.append(ingest_pdf(pdf))
    return results
