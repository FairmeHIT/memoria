# memoria

Deployment and method disclosure:

- [DEPLOYMENT.md](DEPLOYMENT.md)
- [SPEC.md](SPEC.md)
- [BENCHMARK_BASELINES.md](BENCHMARK_BASELINES.md)

Public endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness and readiness |
| `POST` | `/v1/add` | Synchronous memory ingestion |
| `POST` | `/v1/search` | Ranked evidence retrieval |

Quick start:

```bash
docker build --tag memoria:0.4.0 .

docker volume create memoria-data

docker run --rm --publish 8080:8080 \
  --env MEMORIA_API_KEY='<memory-system-key>' \
  --env MEMORIA_AUTH_SCHEME=bearer \
  --env MEMORIA_DATA_DIR=/var/lib/memoria \
  --env MEMORIA_EMBEDDING_BACKEND=none \
  --env MEMORIA_RERANKER_BACKEND=none \
  --mount source=memoria-data,target=/var/lib/memoria \
  memoria:0.4.0
```

Smoke check:

```bash
curl --fail http://localhost:8080/health
```

For the full wrapper examples, deployment profiles, authorship, and method
changes, see [DEPLOYMENT.md](DEPLOYMENT.md).
