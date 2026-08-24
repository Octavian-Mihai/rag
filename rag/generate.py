from __future__ import annotations

import httpx

from rag.config import load_config
from rag.debug_log import debug_log
from rag.models import RetrievedChunk

PROMPT = """You are a legal research assistant. Answer ONLY from the retrieved context.
Cite every factual claim with [doc_id:page] using identifiers from the context list.
If the context is insufficient, say so. Never invent citations.

Question: {question}

Context:
{context}
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        body = (chunk.parent_text or chunk.text)[:1500]
        blocks.append(
            f"[{chunk.doc_id}:{chunk.page}] heading={chunk.heading}\n{body}"
        )
    return "\n\n".join(blocks)


class GenerationError(Exception):
    """Ollama generation failed or timed out."""


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    cfg = load_config()
    host = cfg["ollama"]["host"].rstrip("/")
    model = cfg["models"]["llm"]
    timeout = float(cfg["ollama"]["timeout_s"])
    prompt = PROMPT.format(question=question, context=_format_context(chunks))
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 256},
    }
    # #region agent log
    debug_log(
        "generate.py:generate_answer",
        "ollama_request_start",
        {"host": host, "model": model, "timeout_s": timeout, "prompt_chars": len(prompt)},
        hypothesis_id="H1",
    )
    # #endregion
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{host}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        # #region agent log
        debug_log(
            "generate.py:generate_answer",
            "ollama_timeout",
            {"timeout_s": timeout, "error": str(exc)},
            hypothesis_id="H1",
        )
        # #endregion
        raise GenerationError(
            "Local LLM generation timed out. Ollama may still be loading the model; "
            "try again in a minute or use a shorter question."
        ) from exc
    except httpx.HTTPError as exc:
        # #region agent log
        debug_log(
            "generate.py:generate_answer",
            "ollama_http_error",
            {"error": str(exc)},
            hypothesis_id="H1",
        )
        # #endregion
        raise GenerationError(f"Local LLM unavailable: {exc}") from exc
    answer = str(data.get("response", "")).strip()
    # #region agent log
    debug_log(
        "generate.py:generate_answer",
        "ollama_request_ok",
        {"answer_chars": len(answer)},
        hypothesis_id="H1",
    )
    # #endregion
    return answer
