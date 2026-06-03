from __future__ import annotations

import logging
import numpy as np

from functools import cached_property
from sentence_transformers import SentenceTransformer

from codebase_rag.core.config import Settings, CACHE_MODEL_PATH
from codebase_rag.domain.embedding_port import IEmbeddingPort

logger = logging.getLogger(__name__)

class STEmbedder:
    """
    A Sentence Transformer Embedder that implements back IEmbeddingPort.

    When you use this class, the model is downloaded from HuggingFace on first use and cached locally 
    by the sentence-transformers library. 
    """

    def __init__(self, 
                model_name: str, 
                batch_size: int, 
                device: str):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_path = CACHE_MODEL_PATH

    @classmethod
    def from_settings(cls, setting: Settings) -> STEmbedder:
        return cls(
            model_name=setting.embedding_model_name,
            batch_size=setting.embdding_batch_size,
            device=setting.device
        )

    @cached_property
    def _model(self) -> SentenceTransformer:
        """Lazily loading model then cached it."""
        logger.info("Loading %s on %s", self.model_name, self.device)

        return SentenceTransformer(
            model_name_or_path=self.model_name,
            cache_folder=self.cache_path,
            device=self.device
        )
    
    @cached_property
    def vector_size(self) -> int:
        return self._model.get_embedding_dimension()
    
    def embed_one(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return vectors.tolist()