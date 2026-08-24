from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")
DEBUG_LOG = Path(__file__).resolve().parent.parent / ".cursor" / "debug-8170e8.log"


def _ui_log(message: str, data: dict, hypothesis_id: str = "H3") -> None:
    # #region agent log
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": "8170e8",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": "ui/streamlit_app.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # #endregion


st.set_page_config(page_title="Adaptive Local Legal RAG", layout="wide")
st.title("Adaptive Local Legal RAG")
st.caption("Fully local retrieval: hybrid search, intent routing, and citation grounding.")

with httpx.Client(timeout=10.0) as client:
    try:
        health = client.get(f"{API_BASE}/health").json()
    except Exception as exc:
        health = {"ok": False, "error": str(exc)}

st.sidebar.subheader("Backend")
st.sidebar.json(health)

chunk_count = int(health.get("chunk_count") or 0)
if chunk_count == 0 and health.get("ok"):
    st.info("Indexing sample contracts in the background. Wait a moment, then ask again — or ingest from the sidebar.")

st.sidebar.subheader("Ingest")
uploaded = st.sidebar.file_uploader("Upload a legal PDF", type=["pdf"])
if st.sidebar.button("Ingest upload", disabled=uploaded is None):
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(f"{API_BASE}/ingest", files=files)
    st.sidebar.write(resp.json() if resp.is_success else resp.text)

if st.sidebar.button("Ingest sample contracts"):
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(f"{API_BASE}/ingest", params={"use_samples": True})
    st.sidebar.write(resp.json() if resp.is_success else resp.text)

with st.form("ask_form"):
    question = st.text_input(
        "Question",
        placeholder="What is the payment amount in the Acme service agreement?",
    )
    ask = st.form_submit_button("Ask", type="primary")

if ask and not question.strip():
    st.warning("Enter a question, then press Enter or click Ask.")

if ask and question.strip():
    _ui_log("ui_ask_clicked", {"question_len": len(question.strip()), "api_base": API_BASE})
    st.caption("First answer can take several minutes while Ollama loads the local model.")
    with st.spinner("Retrieving and generating locally (this may take a few minutes)..."):
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(f"{API_BASE}/query", json={"question": question.strip()})
    _ui_log(
        "ui_query_response",
        {"status_code": resp.status_code, "ok": resp.is_success},
        hypothesis_id="H3",
    )
    if not resp.is_success:
        st.error(resp.text)
    else:
        data = resp.json()
        if data.get("abstained"):
            st.warning(data["answer"])
        else:
            st.markdown(data["answer"])
        st.caption(
            f"Route **{data['route']}** · confidence **{data['confidence']:.2f}** · "
            f"{data['latency_ms']:.0f} ms · reason `{data.get('reason')}`"
        )
        st.subheader("Retrieved context")
        for chunk in data.get("chunks", []):
            with st.expander(
                f"{chunk['heading']} · {chunk['doc_id']}:{chunk['page']} · "
                f"score {chunk['score']:.2f} · {chunk['source']}"
            ):
                st.write(chunk["text"])

st.divider()
st.subheader("Metrics")
try:
    with httpx.Client(timeout=10.0) as client:
        st.json(client.get(f"{API_BASE}/metrics").json())
except Exception:
    st.caption("Metrics unavailable until the API is running.")
