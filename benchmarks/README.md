# Latency and resource benchmarks (phase 2)

Default production profile is **BGE-small (384-d)** so Qdrant + Ollama `mistral:7b` + embedder fit ~8GB RAM. BGE-base (768-d) is expected to exceed that budget for a ~3% recall gain.

## Planned measurements
- Query latency by embedding size: 384 vs 768 vs 1024
- Peak RSS for Qdrant + Ollama + BGE (and reranker)
- int8 quantization of embedder/reranker (~50% RAM reduction)

## MVP
`python -m benchmarks.run_latency` records wall-clock for `/health` and a dummy prompt path when the API is up. Full RAM traces are phase 2.
