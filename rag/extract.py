from __future__ import annotations

import re
from pathlib import Path

from rag.models import Chunk

MONEY_RE = re.compile(r"\$\s?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)")
TERM_RE = re.compile(r"(\d+)\s*(?:month|months)", re.IGNORECASE)
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2}|\w+ \d{1,2}, 20\d{2})\b")
PARTY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.\- ]{1,40}(?:Corp(?:oration)?|LLC|Inc\.?|Ltd\.?|LLP))\b"
)


def extract_parties(text: str) -> list[str]:
    found = [m.group(1).strip() for m in PARTY_RE.finditer(text)]
    unique: list[str] = []
    for name in found:
        if name not in unique:
            unique.append(name)
    return unique[:8]


def extract_clause_rows(parents: list[Chunk]) -> list[dict]:
    rows: list[dict] = []
    for parent in parents:
        money = MONEY_RE.search(parent.text)
        term = TERM_RE.search(parent.text)
        date = DATE_RE.search(parent.text)
        parties = extract_parties(parent.text)
        if not (money or term or date or parties):
            continue
        amount = float(money.group(1).replace(",", "")) if money else None
        rows.append(
            {
                "page": parent.page,
                "heading": parent.heading,
                "amount_usd": amount,
                "term_months": int(term.group(1)) if term else None,
                "party": "; ".join(parties) if parties else None,
                "date_iso": date.group(1) if date else None,
            }
        )
    return rows


def title_from_filename(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()
