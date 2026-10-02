# Adaptive Local RAG for Legal Document Intelligence

Production-oriented **local** RAG (no cloud APIs). The MVP routes legal questions to hybrid, graph, or SQL retrieval, reranks with BGE, generates with Ollama Mistral-7B, and abstains on low similarity or ungrounded citations.


## Architecture

```mermaid
flowchart TD
    UI[Streamlit UI :8501] -->|HTTP| API["FastAPI app/main.py"]

    subgraph Ingest["Ingestion"]
        PDF[PDF contracts] --> Ext[extract.py] --> Chunk[chunking/] --> Emb["embedder.py<br/>BGE-small 384-d"]
        Chunk --> Sparse[sparse.py<br/>hashed BM25]
        Chunk --> Meta[(SQLite<br/>clause metadata)]
        Chunk --> Graph[graph.py<br/>document graph]
        Emb --> Q[(Qdrant<br/>dense + sparse)]
        Sparse --> Q
    end

    subgraph Query["Query pipeline — pipeline.py"]
        Router[router.py<br/>intent routing]
        Hyb[retrieve_hybrid]
        Gr[retrieve_graph]
        Sql[retrieve_sql]
        Ens[retrieve_ensemble]
        Rr[rerank.py<br/>BGE reranker]
        Guard[guards.py<br/>abstain on low similarity]
        Gen[generate.py]
        Cit[citations.py<br/>grounding check]
    end

    API --> Ingest
    API --> Router
    Router -->|lookup| Hyb
    Router -->|comparative| Gr
    Router -->|aggregate| Sql
    Router -->|ambiguous| Ens
    Hyb --> Q
    Gr --> Graph
    Sql --> Meta
    Hyb & Gr & Sql & Ens --> Rr --> Guard --> Gen
    Gen <-->|prompt / completion| Ollama[(Ollama<br/>mistral:7b)]
    Gen --> Cit --> API
    API -.-> Metrics[metrics.py]
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Screenshots

| Overview | Query with grounded citations |
| --- | --- |
| ![App overview](docs/screenshots/overview.png) | ![Query answer with retrieved context](docs/screenshots/query_answer.png) |

## Stack
- FastAPI + Streamlit
- Qdrant hybrid search (dense BGE-small 384-d + hashed BM25 sparse)
- Ollama `mistral:7b`
- SQLite clause metadata + in-memory document graph
- Docker Compose

## Quick start
```bash
python scripts/write_sample_pdfs.py
docker compose up --build
```
Wait for `ollama-init` to finish pulling Mistral. Open http://localhost:8501 and click **Ingest sample contracts**.

Without Docker (Qdrant + Ollama already running locally):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
streamlit run ui/streamlit_app.py
```

## Failure modes
- Similarity &lt; 0.45: *I couldn't find relevant information. Please rephrase your query.*
- Router confidence &lt; 0.7: ensemble of hybrid + graph (+ SQL when numeric)
- Generated `[doc_id:page]` not in retrieved chunks: refuse to answer (answers with no citations are still shown)

## Evaluation and performance
See [eval/README.md](eval/README.md) and [benchmarks/README.md](benchmarks/README.md). Default embedder is BGE-small for 8GB-class machines.

```bash
pytest
```
