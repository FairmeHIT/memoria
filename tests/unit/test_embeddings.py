from __future__ import annotations

import math

import pytest

from memoria.embeddings import EmbeddingUnavailable, HashingEmbedder, deserialize_vector, serialize_vector


def test_hashing_embedder_returns_stable_normalized_vectors() -> None:
    embedder = HashingEmbedder(dimensions=32)

    first = embedder.embed(["A developer reviews a pull request."])[0]
    second = embedder.embed(["A developer reviews a pull request."])[0]

    assert first == second
    assert len(first) == 32
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-6)


def test_hashing_embedder_handles_valid_non_token_text() -> None:
    vector = HashingEmbedder(dimensions=32).embed(["!!!"])[0]

    assert math.isclose(sum(value * value for value in vector), 1.0, rel_tol=1e-6)


def test_vector_serialization_rejects_invalid_values_and_dimensions() -> None:
    encoded = serialize_vector((0.6, 0.8))

    assert deserialize_vector(encoded, dimensions=2) == pytest.approx((0.6, 0.8))
    with pytest.raises(EmbeddingUnavailable):
        serialize_vector((float("nan"), 0.0))
    with pytest.raises(EmbeddingUnavailable):
        deserialize_vector(encoded, dimensions=3)
