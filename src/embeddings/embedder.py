import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@dataclass
class EmbeddingCache:
    cache_path: str

    def _load(self) -> dict:
        if not os.path.exists(self.cache_path):
            return {}
        return joblib.load(self.cache_path)

    def _save(self, obj: dict) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        joblib.dump(obj, self.cache_path)

    def get(self, key: str) -> Optional[np.ndarray]:
        store = self._load()
        return store.get(key)

    def set(self, key: str, value: np.ndarray) -> None:
        store = self._load()
        store[key] = value
        self._save(store)


def _hash_texts(texts: Sequence[str]) -> str:
    m = hashlib.sha256()
    for t in texts:
        m.update((t or "").encode("utf-8", errors="replace"))
        m.update(b"\x00")
    return m.hexdigest()


class MPNetEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache: Optional[EmbeddingCache] = None,
    ) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except OSError as e:
            raise RuntimeError(
                "Failed to import sentence-transformers because PyTorch failed to load on Windows. "
                "This is commonly caused by a broken/incorrect torch installation or missing Microsoft Visual C++ "
                "Redistributable (2015-2022, x64).\n\n"
                "Fix (recommended, CPU-only):\n"
                "- Install/repair Microsoft Visual C++ Redistributable 2015-2022 (x64)\n"
                "- Reinstall torch CPU build:\n"
                "  pip uninstall -y torch torchvision torchaudio\n"
                "  pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision\n\n"
                "Original error: "
                + str(e)
            ) from e

        self._model = SentenceTransformer(model_name, device="cpu")
        self._cache = cache

        # Embedding dimensionality is stable for a given model.
        # We infer it once to support zero-vector fallbacks.
        try:
            dim_vec = self._model.encode(
                [""],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            self._dim = int(dim_vec.shape[0])
        except Exception:
            self._dim = 768

    def encode(
        self,
        texts: Sequence[str],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        texts = [t or "" for t in texts]

        cache_key = None
        if self._cache is not None:
            cache_key = f"{self.model_name}::{_hash_texts(texts)}::normalize={normalize}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, emb)

        return emb

    def encode_one(
        self,
        text: str,
        normalize: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        return self.encode(
            [text],
            normalize=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )[0]

    def encode_weighted_chunks(
        self,
        chunks_list: Sequence[Dict[str, str]],
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode chunked documents and pool into one vector per document.

        Pooling rule (default):
            final = 0.4 * summary + 0.4 * experience + 0.2 * skills

        Constraints:
            - Same embedding model
            - If a chunk is empty, its embedding is a zero vector
            - Output has same dimensionality as base model
            - Final embedding is L2-normalized if normalize=True
        """

        if weights is None:
            weights = {"summary": 0.4, "experience": 0.4, "skills": 0.2}

        # Build a deterministic cache key over all chunk texts + weights.
        cache_key = None
        if self._cache is not None:
            key_parts: List[str] = []
            for ch in chunks_list:
                key_parts.append(ch.get("summary", ""))
                key_parts.append(ch.get("experience", ""))
                key_parts.append(ch.get("skills", ""))
            weights_key = f"w={weights.get('summary', 0):.3f},{weights.get('experience', 0):.3f},{weights.get('skills', 0):.3f}"
            cache_key = f"{self.model_name}::chunks::{_hash_texts(key_parts)}::{weights_key}::normalize={normalize}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Flatten all non-empty chunks for a single batched encode.
        flat_texts: List[str] = []
        flat_index: List[Tuple[int, str]] = []
        for i, ch in enumerate(chunks_list):
            for name in ("summary", "experience", "skills"):
                t = (ch.get(name) or "").strip()
                if t:
                    flat_texts.append(t)
                    flat_index.append((i, name))

        # Initialize all to zero vectors.
        out = np.zeros((len(chunks_list), self._dim), dtype=np.float32)

        if flat_texts:
            emb = self._model.encode(
                flat_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            # Add weighted chunk embeddings into the output per document.
            for k, (doc_i, chunk_name) in enumerate(flat_index):
                w = float(weights.get(chunk_name, 0.0))
                if w != 0.0:
                    out[doc_i] += w * emb[k]

        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            out = out / norms

        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, out)

        return out

    def encode_weighted_chunks_one(
        self,
        chunks: Dict[str, str],
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        return self.encode_weighted_chunks(
            [chunks],
            weights=weights,
            normalize=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )[0]


def save_embedder_metadata(path: str, model_name: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"embedding_model": model_name}, f, indent=2)
