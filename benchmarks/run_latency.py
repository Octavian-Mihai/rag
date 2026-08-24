from __future__ import annotations

import argparse
import statistics
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    times: list[float] = []
    with httpx.Client(timeout=30.0) as client:
        for _ in range(args.n):
            start = time.perf_counter()
            client.get(f"{args.base.rstrip('/')}/health").raise_for_status()
            times.append((time.perf_counter() - start) * 1000)
    print(f"health latency_ms n={args.n} mean={statistics.mean(times):.1f} p50={statistics.median(times):.1f}")
    print("Phase 2: swap embedding_dim 384/768/1024 and record peak RSS for Qdrant+Ollama+BGE.")


if __name__ == "__main__":
    main()
