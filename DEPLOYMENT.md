# memoria deployment notes

Repository:

```text
https://github.com/FairmeHIT/memoria
```

## Build

```bash
docker build --tag memoria:0.4.0 .
```

## Run

```bash
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

Container entrypoint:

```bash
uvicorn memoria.app:create_app --factory --host 0.0.0.0 --port 8080
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness and readiness |
| `POST` | `/v1/add` | Ingest ordered messages |
| `POST` | `/v1/search` | Return ranked evidence |

## Add wrapper

```python
import requests

requests.post(
    "https://<host>/v1/add",
    headers={"Authorization": "Bearer <memory-system-key>"},
    json={
        "request_id": "example-add-001",
        "messages": [{"role": "user", "content": "I prefer tea."}],
        "user_id": "example-user-001",
        "session_id": "example-session-001",
    },
    timeout=1200,
).raise_for_status()
```

## Search wrapper

```python
import requests

response = requests.post(
    "https://<host>/v1/search",
    headers={"Authorization": "Bearer <memory-system-key>"},
    json={
        "query": "What do I prefer?",
        "user_id": "example-user-001",
        "top_k": 100,
    },
    timeout=1200,
)
response.raise_for_status()
hits = response.json()["data"]
```

## Smoke check

```bash
curl --fail http://localhost:8080/health
```

If the platform needs a simple end-to-end check:

```bash
curl --fail http://localhost:8080/v1/add \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer <memory-system-key>' \
  --data '{
    "request_id": "smoke-001",
    "messages": [{"role": "user", "content": "I prefer tea."}],
    "user_id": "smoke-user-001",
    "session_id": "smoke-session-001"
  }'
```

```bash
curl --fail http://localhost:8080/v1/search \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer <memory-system-key>' \
  --data '{
    "query": "What do I prefer?",
    "user_id": "smoke-user-001",
    "top_k": 100
  }'
```
