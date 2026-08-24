from __future__ import annotations

from collections import defaultdict

from rag.models import Chunk, RetrievedChunk
from rag.store import all_parent_chunks, list_documents


class DocumentGraph:
    """In-memory section graph: same_doc, next_section, shares_party."""

    def __init__(self) -> None:
        self.nodes: dict[str, Chunk] = {}
        self.edges: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def rebuild(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        parents = all_parent_chunks()
        docs = {d["doc_id"]: d for d in list_documents()}
        by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for node in parents:
            self.nodes[node.chunk_id] = node
            by_doc[node.doc_id].append(node)

        for doc_id, sections in by_doc.items():
            sections.sort(key=lambda c: (c.page, c.heading))
            for section in sections:
                for other in sections:
                    if other.chunk_id != section.chunk_id:
                        self.edges[section.chunk_id].append(("same_doc", other.chunk_id))
            for left, right in zip(sections, sections[1:]):
                self.edges[left.chunk_id].append(("next_section", right.chunk_id))

        party_index: dict[str, list[str]] = defaultdict(list)
        for doc_id, meta in docs.items():
            parties = [p.strip().lower() for p in (meta.get("parties") or "").split(";") if p.strip()]
            for party in parties:
                for section in by_doc.get(doc_id, []):
                    party_index[party].append(section.chunk_id)
        for ids in party_index.values():
            unique = list(dict.fromkeys(ids))
            for src in unique:
                for dst in unique:
                    if src != dst and self.nodes[src].doc_id != self.nodes[dst].doc_id:
                        self.edges[src].append(("shares_party", dst))

    def neighbors(self, chunk_id: str, hops: int = 1) -> list[Chunk]:
        seen = {chunk_id}
        frontier = [chunk_id]
        found: list[Chunk] = []
        for _ in range(hops):
            nxt: list[str] = []
            for node_id in frontier:
                for _rel, dest in self.edges.get(node_id, []):
                    if dest not in seen:
                        seen.add(dest)
                        nxt.append(dest)
                        node = self.nodes.get(dest)
                        if node:
                            found.append(node)
            frontier = nxt
        return found

    def as_retrieved(self, chunks: list[Chunk], score: float = 0.75) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                parent_id=c.parent_id,
                page=c.page,
                text=c.text,
                heading=c.heading,
                score=score,
                parent_text=c.text,
                source="graph",
            )
            for c in chunks
        ]


GRAPH = DocumentGraph()
