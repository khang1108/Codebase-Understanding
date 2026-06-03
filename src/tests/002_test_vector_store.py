"""Unit tests for FaissStore vector store implementation.

Tests cover all public methods of FaissStore using small fixed-dimension
vectors. No embedding model is loaded — vectors are hand-crafted so that
expected search rankings are predictable without semantic meaning.

Vector dimension is fixed at 4 for speed. All test vectors are
L2-normalised (unit length) to match the real pipeline contract.
"""

from __future__ import annotations

import pytest

from codebase_rag.domain.code_chunk import CodeChunk
from codebase_rag.domain.vector_store import SearchFilters
from codebase_rag.infrastructure.vectordb.faiss import FaissStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_SIZE = 4

# Pre-normalised unit vectors — each is orthogonal to the others.
# IndexFlatIP score = dot product. For unit vectors, dot product equals cosine.
VEC_A = [1.0, 0.0, 0.0, 0.0]   # "auth" direction
VEC_B = [0.0, 1.0, 0.0, 0.0]   # "payment" direction
VEC_C = [0.0, 0.0, 1.0, 0.0]   # "config" direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(
    chunk_id: str = "src/auth/login.py:1-30",
    file_path: str = "src/auth/login.py",
    language: str = "python",
    text: str = "def login(): ...",
    repo_name: str | None = None,
) -> CodeChunk:
    """Return a CodeChunk with sensible defaults for testing."""
    return CodeChunk(
        chunk_id=chunk_id,
        file_path=file_path,
        language=language,
        start_line=1,
        end_line=30,
        text=text,
        repo_name=repo_name,
    )


@pytest.fixture
def store() -> FaissStore:
    """Return an initialised FaissStore ready for upsert."""
    s = FaissStore(vector_size=VECTOR_SIZE)
    s.create_collection()
    return s


# ---------------------------------------------------------------------------
# create_collection
# ---------------------------------------------------------------------------

class TestCreateCollection:

    def test_creates_empty_index(self):
        s = FaissStore(vector_size=VECTOR_SIZE)
        s.create_collection()
        assert s._index is not None
        assert s._index.ntotal == 0

    def test_clears_chunk_list(self):
        s = FaissStore(vector_size=VECTOR_SIZE)
        s.create_collection()
        assert s._chunks == []

    def test_idempotent_does_not_overwrite(self, store):
        """Calling create_collection twice must not wipe existing data."""
        store.upsert([make_chunk()], [VEC_A])
        store.create_collection()           # second call
        assert store._index.ntotal == 1     # data still there

    def test_raises_on_invalid_vector_size(self):
        s = FaissStore(vector_size=0)
        with pytest.raises(ValueError, match="vector_size"):
            s.create_collection()


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------

class TestUpsert:

    def test_adds_chunks_to_index(self, store):
        chunks = [make_chunk("id1"), make_chunk("id2", file_path="b.py")]
        store.upsert(chunks, [VEC_A, VEC_B])
        assert store._index.ntotal == 2
        assert len(store._chunks) == 2

    def test_raises_if_not_initialised(self):
        s = FaissStore(vector_size=VECTOR_SIZE)
        with pytest.raises(RuntimeError, match="create_collection"):
            s.upsert([make_chunk()], [VEC_A])

    def test_raises_on_length_mismatch(self, store):
        with pytest.raises(ValueError, match="equal length"):
            store.upsert([make_chunk()], [VEC_A, VEC_B])

    def test_skips_duplicate_when_overwrite_false(self, store, caplog):
        chunk = make_chunk("dup_id")
        store.upsert([chunk], [VEC_A])
        store.upsert([chunk], [VEC_B], overwrite=False)
        # index should still contain only 1 vector
        assert store._index.ntotal == 1

    def test_overwrites_duplicate_when_overwrite_true(self, store):
        chunk_v1 = make_chunk("dup_id", text="version 1")
        chunk_v2 = make_chunk("dup_id", text="version 2")
        store.upsert([chunk_v1], [VEC_A])
        store.upsert([chunk_v2], [VEC_B], overwrite=True)
        # metadata updated
        assert store._chunks[0].text == "version 2"
        # index size unchanged
        assert store._index.ntotal == 1

    def test_mixed_new_and_duplicate(self, store):
        chunk_a = make_chunk("id_a")
        chunk_b = make_chunk("id_b", file_path="b.py")
        store.upsert([chunk_a], [VEC_A])
        # id_a is duplicate, id_b is new
        store.upsert([chunk_a, chunk_b], [VEC_B, VEC_C], overwrite=False)
        assert store._index.ntotal == 2     # only id_b was added


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:

    def test_returns_top_k_results(self, store):
        chunks = [make_chunk(f"id{i}", file_path=f"f{i}.py") for i in range(5)]
        vectors = [VEC_A, VEC_B, VEC_C, VEC_B, VEC_A]
        store.upsert(chunks, vectors)
        results = store.search(VEC_A, top_k=3)
        assert len(results) == 3

    def test_results_ordered_by_descending_score(self, store):
        store.upsert(
            [make_chunk("auth", file_path="auth.py"), make_chunk("pay", file_path="pay.py")],
            [VEC_A, VEC_B],
        )
        # query close to VEC_A — auth chunk should rank first
        results = store.search(VEC_A, top_k=2)
        assert results[0].score >= results[1].score

    def test_closest_vector_scores_highest(self, store):
        store.upsert(
            [make_chunk("auth", file_path="auth.py"), make_chunk("pay", file_path="pay.py")],
            [VEC_A, VEC_B],
        )
        results = store.search(VEC_A, top_k=2)
        assert results[0].file_path == "auth.py"

    def test_result_fields_populated(self, store):
        chunk = make_chunk(
            chunk_id="src/auth/login.py:1-30",
            file_path="src/auth/login.py",
            language="python",
            text="def login(): ...",
            repo_name="my-repo",
        )
        store.upsert([chunk], [VEC_A])
        result = store.search(VEC_A, top_k=1)[0]
        assert result.chunk_id == chunk.chunk_id
        assert result.file_path == chunk.file_path
        assert result.language == chunk.language
        assert result.start_line == chunk.start_line
        assert result.end_line == chunk.end_line
        assert result.text == chunk.text
        assert result.repo_name == chunk.repo_name
        assert 0.0 <= result.score <= 1.0

    def test_filter_by_language(self, store):
        store.upsert(
            [
                make_chunk("py_chunk", language="python"),
                make_chunk("js_chunk", file_path="app.js", language="javascript"),
            ],
            [VEC_A, VEC_A],
        )
        results = store.search(VEC_A, top_k=10, filters=SearchFilters(language="python"))
        assert all(r.language == "python" for r in results)
        assert len(results) == 1

    def test_filter_by_file_path(self, store):
        store.upsert(
            [
                make_chunk("a", file_path="auth/login.py"),
                make_chunk("b", file_path="payment/service.py"),
            ],
            [VEC_A, VEC_A],
        )
        results = store.search(
            VEC_A, top_k=10, filters=SearchFilters(file_path="auth/login.py")
        )
        assert len(results) == 1
        assert results[0].file_path == "auth/login.py"

    def test_returns_empty_when_no_filter_match(self, store):
        store.upsert([make_chunk(language="python")], [VEC_A])
        results = store.search(
            VEC_A, top_k=5, filters=SearchFilters(language="java")
        )
        assert results == []

    def test_raises_if_not_initialised(self):
        s = FaissStore(vector_size=VECTOR_SIZE)
        with pytest.raises(RuntimeError, match="create_collection"):
            s.search(VEC_A, top_k=5)

    def test_raises_on_wrong_vector_dimension(self, store):
        store.upsert([make_chunk()], [VEC_A])
        with pytest.raises(ValueError, match="vector_size"):
            store.search([1.0, 0.0], top_k=1)   # dim=2, not 4

    def test_raises_on_invalid_top_k(self, store):
        store.upsert([make_chunk()], [VEC_A])
        with pytest.raises(ValueError, match="top_k"):
            store.search(VEC_A, top_k=0)


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------

class TestDeleteCollection:

    def test_resets_index(self, store):
        store.upsert([make_chunk()], [VEC_A])
        store.delete_collection()
        assert store._index is None

    def test_resets_chunks(self, store):
        store.upsert([make_chunk()], [VEC_A])
        store.delete_collection()
        assert store._chunks == []

    def test_search_raises_after_delete(self, store):
        store.delete_collection()
        with pytest.raises(RuntimeError, match="create_collection"):
            store.search(VEC_A, top_k=1)

    def test_can_recreate_after_delete(self, store):
        store.delete_collection()
        store.create_collection()
        store.upsert([make_chunk()], [VEC_A])
        assert store._index.ntotal == 1


# ---------------------------------------------------------------------------
# _matches (private helper)
# ---------------------------------------------------------------------------

class TestMatches:

    def test_matches_all_conditions(self):
        s = FaissStore(vector_size=VECTOR_SIZE)
        chunk = make_chunk(language="python", file_path="auth/login.py", repo_name="my-repo")
        assert s._matches(chunk, SearchFilters(language="python", file_path="auth/login.py"))

    def test_fails_on_one_mismatch(self):
        s = FaissStore(vector_size=VECTOR_SIZE)
        chunk = make_chunk(language="python")
        assert not s._matches(chunk, SearchFilters(language="java"))

    def test_empty_filters_always_matches(self):
        s = FaissStore(vector_size=VECTOR_SIZE)
        chunk = make_chunk()
        assert s._matches(chunk, SearchFilters())
