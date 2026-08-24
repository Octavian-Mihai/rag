from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parent.parent / ".cursor" / "debug-8170e8.log"
SESSION_ID = "8170e8"


def debug_log(
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    *,
    hypothesis_id: str = "",
    run_id: str = "pre-fix",
) -> None:
    # #region agent log
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # #endregion
