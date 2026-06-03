from __future__ import annotations

import math
import pytest

from codebase_rag.infrastructure.embedding.sentence_transformers_embedder import STEmbedder
from codebase_rag.domain.embedding_port import IEmbeddingPort

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXPECTED_DIM = 384

@pytest.fixture(scope="module")
def embedder() -> STEmbedder:
    return STEmbedder(
        model_name=MODEL_NAME,
        batch_size=32,
        device="cpu"
    )

def test_satisfies_protocol(embedder):
    assert isinstance(embedder, IEmbeddingPort)

def test_vector_size(embedder):
    assert embedder.vector_size == EXPECTED_DIM

def test_embed_batch_shape(embedder):
    texts = ["def login():", "class UserService:", "SELECT * FROM users"]
    vectors = embedder.embed_batch(texts)

    assert len(vectors) == 3
    assert all(len(v) == EXPECTED_DIM for v in vectors)

def test_embed_batch_normalised(embedder):
    """Vectors must be L2-normalised — each vector's norm must be ≈ 1.0."""
    vectors = embedder.embed_batch(["hello world"])
    norm = math.sqrt(sum(x ** 2 for x in vectors[0]))
    assert abs(norm - 1.0) < 1e-5

def test_embed_one_shape(embedder):
    vector = embedder.embed_one("where is login handled?")
    assert len(vector) == EXPECTED_DIM

def test_embed_one_delegates_to_embed_batch(embedder):
    """embed_one and embed_batch must return the same vector for the same input."""
    text = "def authenticate(user, password):"
    assert embedder.embed_one(text) == embedder.embed_batch([text])[0]

def test_similar_texts_score_higher_than_unrelated(embedder):
    """Semantic sanity check — related texts must score higher than unrelated ones."""
    query = embedder.embed_one("user login authentication")
    related = embedder.embed_one("def login(username, password):")
    unrelated = embedder.embed_one("SELECT * FROM products WHERE price > 100")

    score_related = sum(q * r for q, r in zip(query, related))
    score_unrelated = sum(q * r for q, r in zip(query, unrelated))

    assert score_related > score_unrelated