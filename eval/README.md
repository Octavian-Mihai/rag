# Evaluation framework (phase 2)

## Golden dataset
- Schema: JSONL with `query`, `relevant_contains`, `intent`
- MVP: 10 labeled pairs in `golden.jsonl` derived from sample contracts
- Target: 50 hand-curated query–document pairs

## Synthetic queries
- Generate 200+ queries with RAGAS (open-source) over ingested parents
- Keep all generation local (no cloud LLM APIs)

## Metrics
- Hit Rate@5 and MRR: implemented in `metrics.py` / `run_retrieval.py`
- Context Relevancy and Faithfulness: RAGAS stubs (`NotImplementedError` until phase 2)

## Bake-off (phase 2)
Compare chunking: fixed-size vs semantic parent-child vs sliding-window
(`rag/chunking/__init__.py` already includes all three helpers).

Compare embeddings: BGE-small (384), BGE-base (768), all-MiniLM-L6-v2.
Output a winner table with justification (recall vs RAM).

## Run (after ingest)
```bash
python -m eval.run_retrieval
```
