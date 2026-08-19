"""Local embedding interfaces and safe vector serialization."""
from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from memoria.model_audit import ModelCallAudit


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


class EmbeddingUnavailable(RuntimeError):
    """An enabled embedding channel could not produce a valid vector."""


class EmbeddingProvider(Protocol):
    """A deterministic local text-to-vector implementation."""

    fingerprint: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class ModelCallRecorder(Protocol):
    def record(self, audit: ModelCallAudit) -> None: ...


class HashingEmbedder:
    """Offline dense lexical vectors used when no learned model is configured."""

    fingerprint: str

    def __init__(self, *, dimensions: int) -> None:
        if dimensions < 8:
            raise ValueError("embedding dimensions must be at least 8")
        self.dimensions = dimensions
        self.fingerprint = f"hashing-charword-v1-{dimensions}"

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self.dimensions
        tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
        features = list(tokens)
        features.extend(f"{left}|{right}" for left, right in zip(tokens, tokens[1:], strict=False))
        for token in tokens:
            padded = f"^{token}$"
            features.extend(padded[index : index + 3] for index in range(max(len(padded) - 2, 0)))
        if not features:
            features.append(f"raw:{text.casefold()}")
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            values[index] += 1.0 if digest[4] & 1 else -1.0
        magnitude = math.sqrt(sum(value * value for value in values))
        if magnitude == 0.0:
            return tuple(values)
        return tuple(value / magnitude for value in values)


_BGE_MODEL_URLS = (
    "https://hf-mirror.com/{repo_id}/resolve/main/{filename}",
    "https://huggingface.co/{repo_id}/resolve/main/{filename}",
)


def _auto_download_model(target_dir: str | Path) -> Path:
    """Download the BGE model weights to *target_dir* if it is empty.

    Tries the Hugging Face mirror (hf-mirror.com) first, then the official
    endpoint.  Raises ``RuntimeError`` if neither is reachable.
    """
    dest = Path(target_dir)
    if dest.exists() and any(dest.iterdir()):
        return dest  # already populated

    dest.mkdir(parents=True, exist_ok=True)
    repo_id = "BAAI/bge-small-en-v1.5"
    # The minimal set of files SentenceTransformer needs to run.
    required = (
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "model.safetensors",
    )

    import urllib.request as _request

    for url_template in _BGE_MODEL_URLS:
        ok = 0
        for name in required:
            url = url_template.format(repo_id=repo_id, filename=name)
            out = dest / name
            try:
                _request.urlretrieve(url, out, timeout=120)
                ok += 1
            except Exception:
                break
        if ok == len(required):
            return dest
    raise RuntimeError(
        f"Could not download model {repo_id} from any mirror. "
        "Please check your internet connection, or install the model manually: "
        "https://hf-mirror.com/{repo_id}"
    )


class LocalBgeEmbedder:
    """Local sentence-transformers BGE embedding (CPU, no external service.

    Mirrors InvMem's approach: bge-small-en-v1.5 loaded locally via
    sentence-transformers, with BGE query prefix. Runs on CPU by default,
    requires ~130 MB memory and ~5-10 ms per message.

    model_name can be a HuggingFace repo ID (e.g. "BAAI/bge-small-en-v1.5")
    or a local path (e.g. "models/bge-small-en-v1.5"). When the path does not
    exist yet, the model is downloaded automatically — first from the Hugging
    Face mirror (hf-mirror.com, fast in CN) and then from huggingface.co.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimensions: int = 384,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        # Normalize the model reference: if it has a "/" it is a repo ID
        # (or an explicit path); otherwise treat bare names as repo IDs.
        repo_id = model_name
        local_dir: Path | None = None
        candidate = Path(model_name)
        if not model_name.startswith(("./", "../", "/")) and candidate.exists():
            # A concrete existing path: keep it, and base the fingerprint on
            # the basename so a moved checkout still reuses cached vectors.
            local_dir = candidate
            repo_id = candidate.name
        elif "/" in model_name and candidate.exists():
            local_dir = candidate
            repo_id = candidate.name
        # If a local path was requested but is missing, download it first.
        if local_dir is None and model_name.startswith(("./", "../", "/")):
            local_dir = _auto_download_model(model_name)
            repo_id = Path(model_name).name
        if local_dir is not None:
            model_path = str(local_dir.resolve())
        else:
            # Repo ID: check conventional checkout dirs before falling
            # through to sentence-transformers (which downloads from HF).
            for candidate_dir in (
                Path("models") / repo_id,                     # models/BAAI/bge-small-en-v1.5
                Path("models") / repo_id.rsplit("/", 1)[-1],  # models/bge-small-en-v1.5
                Path("models") / repo_id.replace("/", "--"),  # models/BAAI--bge-small-en-v1.5
            ):
                if candidate_dir.exists():
                    model_path = str(candidate_dir.resolve())
                    break
            else:
                model_path = repo_id  # download via sentence-transformers

        self.model_name = repo_id
        self.dimensions = dimensions
        # Fingerprint on the bare model name so the default repo ID
        # (BAAI/bge-small-en-v1.5) and a local checkout (models/bge-small-en-v1.5)
        # share the same vector cache fingerprint.
        self.fingerprint = f"local-bge:{Path(repo_id).name}:{dimensions}"
        # BGE query prefix: marks the query side for asymmetric retrieval.
        # InvMem applies this automatically when model name contains "bge" and "-en-".
        self._query_prefix = "Represent this sentence for searching relevant passages: "
        self._model = SentenceTransformer(model_path)

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._encode_one(text) for text in texts)

    def _encode_one(self, text: str) -> tuple[float, ...]:
        vectors = self._model.encode(
            [self._query_prefix + text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return tuple(float(v) for v in vectors[0])


def create_embedder(
    *,
    backend: str,
    dimensions: int,
    api_key: str | None = None,
    base_url: str = "",
    model: str = "text-embedding-v4",
    timeout_seconds: float = 15.0,
    retries: int = 2,
    recorder: ModelCallRecorder | None = None,
    auth_scheme: str = "bearer",
) -> EmbeddingProvider | None:
    """Create a configured local, Qwen, or self-hosted BGE provider."""

    if backend == "none":
        return None
    if backend == "hashing":
        return HashingEmbedder(dimensions=dimensions)
    if backend == "qwen":
        from memoria.qwen import QwenEmbeddingProvider

        if api_key is None:
            raise ValueError("Qwen embedding API key is required")
        return QwenEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
            retries=retries,
            recorder=recorder,
        )
    if backend == "bge":
        from memoria.bge import BgeEmbeddingProvider

        if api_key is None:
            raise ValueError("BGE embedding API key is required")
        return BgeEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
            retries=retries,
            auth_scheme=auth_scheme,
            recorder=recorder,
        )
    if backend == "local":
        return LocalBgeEmbedder(model_name=model, dimensions=dimensions)
    raise ValueError("MEMORIA_EMBEDDING_BACKEND must be none, hashing, qwen, bge, or local")


class LocalCrossEncoder:
    """Local CrossEncoder reranker via sentence-transformers (CPU, no external service).

    Mirrors InvMem's approach: cross-encoder/ms-marco-MiniLM-L-6-v2 loaded
    locally via sentence-transformers CrossEncoder. Takes a query and a list of
    candidate documents, scores each (query, document) pair, and returns
    ranked indices with scores. Runs on CPU, ~80 MB memory, ~1-2 ms per pair.

    model_name can be a HuggingFace repo ID or a local path. When the path
    does not exist, the model is downloaded automatically from the Hugging
    Face mirror first.
    """

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.fingerprint = f"local-ce:{model_name}"
        # Resolve the model path like LocalBgeEmbedder does.
        repo_id = model_name
        local_dir: Path | None = None
        candidate = Path(model_name)
        if not model_name.startswith(("./", "../", "/")) and candidate.exists():
            local_dir = candidate
        elif "/" in model_name and candidate.exists():
            local_dir = candidate
        if local_dir is None and model_name.startswith(("./", "../", "/")):
            local_dir = _auto_download_model(model_name)
        if local_dir is not None:
            model_path = str(local_dir.resolve())
        else:
            for candidate_dir in (
                Path("models") / repo_id,
                Path("models") / repo_id.rsplit("/", 1)[-1],
                Path("models") / repo_id.replace("/", "--"),
            ):
                if candidate_dir.exists():
                    model_path = str(candidate_dir.resolve())
                    break
            else:
                model_path = repo_id
        self._model = CrossEncoder(model_path)

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> tuple[tuple[int, float], ...]:
        if not documents:
            return ()
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        # CrossEncoder returns raw scores; higher is better.
        indexed = sorted(
            enumerate(scores.tolist() if hasattr(scores, "tolist") else scores),
            key=lambda x: x[1],
            reverse=True,
        )
        if top_n is not None:
            indexed = indexed[:top_n]
        return tuple((idx, float(score)) for idx, score in indexed)


def serialize_vector(vector: Sequence[float]) -> bytes:
    """Encode an L2-normalized vector as explicit little-endian float32 values."""

    values = tuple(float(value) for value in vector)
    if not values or any(not math.isfinite(value) for value in values):
        raise EmbeddingUnavailable("embedding provider returned an invalid vector")
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0.0:
        raise EmbeddingUnavailable("embedding provider returned a zero vector")
    normalized = tuple(value / magnitude for value in values)
    return struct.pack(f"<{len(normalized)}f", *normalized)


def deserialize_vector(payload: bytes, *, dimensions: int) -> tuple[float, ...]:
    """Decode a stored vector after validating its immutable dimensions."""

    expected_size = dimensions * 4
    if dimensions < 1 or len(payload) != expected_size:
        raise EmbeddingUnavailable("stored embedding has an invalid dimension")
    values = struct.unpack(f"<{dimensions}f", payload)
    if any(not math.isfinite(value) for value in values):
        raise EmbeddingUnavailable("stored embedding contains a non-finite value")
    return tuple(values)
