from __future__ import annotations

import httpx

from rag.config import load_config
from rag.debug_log import debug_log


def warmup_ollama() -> None:
    cfg = load_config()
    host = cfg["ollama"]["host"].rstrip("/")
    model = cfg["models"]["llm"]
    payload = {
        "model": model,
        "prompt": "Reply with OK",
        "stream": False,
        "options": {"num_predict": 2, "temperature": 0},
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            client.post(f"{host}/api/generate", json=payload)
        # #region agent log
        debug_log("warmup.py:warmup_ollama", "ollama_warmup_ok", {"model": model}, hypothesis_id="H1")
        # #endregion
    except Exception as exc:
        # #region agent log
        debug_log("warmup.py:warmup_ollama", "ollama_warmup_failed", {"error": str(exc)}, hypothesis_id="H1")
        # #endregion
