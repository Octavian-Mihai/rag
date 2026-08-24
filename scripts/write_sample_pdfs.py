from __future__ import annotations

from pathlib import Path


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap(text: str, width: int = 90) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw:
            lines.append("")
            continue
        words = raw.split()
        current: list[str] = []
        for word in words:
            trial = " ".join(current + [word])
            if current and len(trial) > width:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
    return lines


def write_pdf(path: Path, pages: list[str]) -> None:
    objects: list[bytes] = []

    def add(payload: str) -> int:
        objects.append(payload.encode("latin-1", errors="replace"))
        return len(objects)

    font_id = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    content_ids: list[int] = []
    for page_text in pages:
        commands = ["BT", "/F1 11 Tf"]
        y = 720
        for raw_line in _wrap(page_text):
            commands.append(f"1 0 0 1 50 {y} Tm ({_escape(raw_line)}) Tj")
            y -= 14
            if y < 60:
                break
        commands.append("ET")
        stream = "\n".join(commands)
        body = stream.encode("latin-1", errors="replace")
        content_ids.append(add(f"<< /Length {len(body)} >>\nstream\n{stream}\nendstream"))

    page_id_start = len(objects) + 2
    kids = " ".join(f"{page_id_start + i} 0 R" for i in range(len(pages)))
    pages_id = add(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    for content_id in content_ids:
        add(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf.extend(f"{i} 0 obj\n".encode())
        buf.extend(obj)
        buf.extend(b"\nendobj\n")
    xref_at = len(buf)
    buf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    buf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode())
    buf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)


SERVICE = """MASTER SERVICES AGREEMENT

This Master Services Agreement is entered into as of January 15, 2024 between Acme Corp and Beta LLC.

ARTICLE 1 PARTIES
Acme Corp is the client. Beta LLC is the service provider. Notices to Acme Corp shall be sent to 100 Market Street.

ARTICLE 2 PAYMENT
Beta LLC shall invoice Acme Corp $50,000 per year. Invoices are due within 30 days. Late amounts accrue interest.

ARTICLE 3 TERM
The initial term is 12 months beginning January 15, 2024 and renews annually unless either party gives 60 days notice.

ARTICLE 4 TERMINATION
Either party may terminate for material breach if the breach remains uncured for 30 days after written notice.

ARTICLE 5 LIABILITY
The liability cap is $100,000 except for fraud or willful misconduct. Indirect damages are excluded.

ARTICLE 6 GOVERNING LAW
This agreement is governed by the laws of the State of New York.
"""

NDA = """MUTUAL NONDISCLOSURE AGREEMENT

This Mutual Nondisclosure Agreement is entered into as of March 1, 2024 between Acme Corp and Gamma Inc.

ARTICLE 1 PARTIES
Acme Corp and Gamma Inc agree to protect Confidential Information exchanged for evaluating a potential partnership.

ARTICLE 2 CONFIDENTIALITY
Confidential Information includes technical, financial, and customer data. Gamma Inc shall not disclose Acme Corp secrets.

ARTICLE 3 TERM
Confidentiality obligations last 24 months from March 1, 2024.

ARTICLE 4 PAYMENT
No license fee is due. Each party bears its own costs. A $5,000 administrative deposit may be required.

ARTICLE 5 LIABILITY
Liability is limited to $25,000 for unauthorized disclosure, excluding willful misconduct.

ARTICLE 6 GOVERNING LAW
This agreement is governed by the laws of the State of Delaware.
"""

EMPLOYMENT = """EMPLOYMENT AGREEMENT

This Employment Agreement is entered into as of June 1, 2024 between Acme Corp and Delta LLP as employer of record.

ARTICLE 1 PARTIES
Acme Corp engages the employee through Delta LLP. The workplace is New York.

ARTICLE 2 PAYMENT
Base compensation is $120,000 per year, paid monthly. A signing bonus of $10,000 is due within 30 days.

ARTICLE 3 TERM
The employment term is 36 months from June 1, 2024 unless terminated earlier.

ARTICLE 4 TERMINATION
Acme Corp may terminate without cause with 14 days notice. For cause termination is immediate.

ARTICLE 5 LIABILITY
Indemnification by Acme Corp is capped at $75,000 for third-party claims arising from assigned work.

ARTICLE 6 GOVERNING LAW
This agreement is governed by the laws of the State of New York.
"""


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "samples"
    write_pdf(out / "acme_beta_service_agreement.pdf", [SERVICE])
    write_pdf(out / "acme_gamma_nda.pdf", [NDA])
    write_pdf(out / "acme_delta_employment.pdf", [EMPLOYMENT])
    print(f"Wrote sample PDFs to {out}")


if __name__ == "__main__":
    main()
