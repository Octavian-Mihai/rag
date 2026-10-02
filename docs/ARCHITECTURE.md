# Architecture

Local, adaptive RAG for legal documents: route → retrieve → rerank → guard → generate → verify citations.

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

Deployed with Docker Compose (`Dockerfile.api`, `Dockerfile.ui`, Qdrant, Ollama). `eval/` and `benchmarks/` measure retrieval quality and latency.
