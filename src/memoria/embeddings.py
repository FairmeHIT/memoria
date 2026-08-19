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


# 中文 CJK 正则，用于双模型路由判断
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

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

    instruction (optional): prefix prepended to the query (e.g. for BGE
    reranker models that require a task instruction).
    """

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        instruction: str = "",
    ) -> None:
        self.model_name = model_name
        self.instruction = instruction
        self.fingerprint = f"local-ce:{model_name}"
        # 纯英文模型（ms-marco）不支持中文查询；多语言模型不受限
        self.supports_cjk = "ms-marco" not in model_name
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
        # num_labels=1 的模型（如 BGE reranker v2-m3）需要在 logits 空间
        # 排序；sigmoid 会压缩分数导致难以区分。用 transformers 直接推理。
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path)
            use_raw_logits = int(getattr(config, "num_labels", 1)) == 1
        except Exception:
            use_raw_logits = False
        self._use_raw_logits = use_raw_logits
        if use_raw_logits:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._rf_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._rf_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = None  # type: ignore[assignment]
        else:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_path)
            self._rf_model = None  # type: ignore[assignment]
            self._rf_tokenizer = None  # type: ignore[assignment]

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> tuple[tuple[int, float], ...]:
        if not documents:
            return ()
        q = f"{self.instruction} {query}".strip() if self.instruction else query
        if self._use_raw_logits:
            import torch

            pairs = [(q, doc) for doc in documents]
            inputs = self._rf_tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self._rf_model(**inputs).logits
            if logits.shape[-1] == 1:
                scores = logits.squeeze(-1).tolist()
            else:
                scores = logits[:, 0].tolist()
            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            if top_n is not None:
                indexed = indexed[:top_n]
            return tuple((idx, float(score)) for idx, score in indexed)
        pairs = [(q, doc) for doc in documents]
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


class RoutingCrossEncoder:
    """按查询语言路由的 CrossEncoder 组合。

    英文/拉丁查询交给 ms-marco（对记忆治理、偏好等维度得分稳定），
    中文等 CJK 查询交给多语言模型（bge-reranker-v2-m3），两者共享
    相同的 rerank() 接口。score 空间因模型不同而异，调用方应做
    min-max 归一化（store._apply_reranker 已处理）。
    """

    def __init__(
        self,
        *,
        english_model: LocalCrossEncoder,
        multilingual_model: LocalCrossEncoder,
    ) -> None:
        self._english = english_model
        self._multilingual = multilingual_model
        self.fingerprint = f"route:{english_model.fingerprint}|{multilingual_model.fingerprint}"
        self.supports_cjk = True

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> tuple[tuple[int, float], ...]:
        if _CJK_RE.search(query):
            return self._multilingual.rerank(query, documents, top_n=top_n)
        return self._english.rerank(query, documents, top_n=top_n)


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
