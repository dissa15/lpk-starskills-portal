"""Metode retrieval untuk RAG: BM25 (tanpa dependensi berat), dense (embedding + ChromaDB), hybrid.

Setiap retriever punya `search(query, k)` yang mengembalikan daftar `Hit` terurut.
`Hit.score` dipakai untuk menilai apakah informasinya ADA di KB (dibandingkan dengan `min_score`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from .knowledge import Entry

STOP = set("""yang dan di ke dari untuk dengan atau ini itu pada adalah ada apa apakah saya kami kita anda
bisa dapat akan juga sudah belum tidak boleh mau ingin dong sih ya kak saja aja nya berapa bagaimana
bagaimanakah kah lah oleh agar sebagai dalam para serta jika kalau yaitu tersebut""".split())


def tokenize(text: str) -> List[str]:
    t = re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())
    return [w for w in t.split() if w not in STOP]


@dataclass(frozen=True)
class Hit:
    entry: Entry
    score: float


class BM25Retriever:
    """BM25 Okapi (implementasi sendiri). Skor tidak punya skala tetap, ambang perlu dikalibrasi."""
    kind = 'bm25'

    def __init__(self, entries: Sequence[Entry], min_score: float, k1: float = 1.5, b: float = 0.75):
        self.entries = list(entries)
        self.min_score, self.k1, self.b = min_score, k1, b
        self.docs = [tokenize(e.text) for e in self.entries]
        self.N = len(self.docs)
        self.avgdl = float(np.mean([len(d) for d in self.docs]))
        df: dict = {}
        for d in self.docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for w, n in df.items()}
        self.tf = [{w: d.count(w) for w in set(d)} for d in self.docs]

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.N)
        q = tokenize(query)
        for i, d in enumerate(self.docs):
            dl = len(d)
            for w in q:
                f = self.tf[i].get(w)
                if f:
                    out[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out

    def search(self, query: str, k: int = 3) -> List[Hit]:
        s = self.scores(query)
        order = np.argsort(-s)[:k]
        return [Hit(self.entries[i], float(s[i])) for i in order]


def default_embed_fn(model_name: str) -> Callable[[List[str]], List[List[float]]]:
    """Embedding Sentence Transformers (diimpor saat dibutuhkan, karena berat)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def embed(texts: List[str]) -> List[List[float]]:
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    return embed


class DenseRetriever:
    """Dense retrieval: embedding + ChromaDB (cosine). Skor = kemiripan kosinus (0 sampai 1)."""
    kind = 'dense'

    def __init__(self, entries: Sequence[Entry], min_score: float,
                 embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 persist_dir: Optional[str] = None, collection_name: str = 'kb_lpk'):
        import chromadb
        self.entries = list(entries)
        self.min_score = min_score
        self.embed = embed_fn or default_embed_fn(model_name)
        client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.Client()
        try:  # bangun ulang agar selalu sinkron dengan CSV
            client.delete_collection(collection_name)
        except Exception:
            pass
        self.col = client.create_collection(name=collection_name, metadata={'hnsw:space': 'cosine'})
        self.col.add(
            ids=[e.id for e in self.entries],
            documents=[e.text for e in self.entries],
            embeddings=self.embed([e.text for e in self.entries]),
            metadatas=[{'kategori': e.kategori, 'status': e.status} for e in self.entries],
        )
        self._by_id = {e.id: e for e in self.entries}

    def search(self, query: str, k: int = 3) -> List[Hit]:
        r = self.col.query(query_embeddings=self.embed([query]), n_results=min(k, len(self.entries)))
        return [Hit(self._by_id[i], float(1.0 - d)) for i, d in zip(r['ids'][0], r['distances'][0])]


class HybridRetriever:
    """Gabungan BM25 + dense dengan Reciprocal Rank Fusion.

    Urutan hasil dari RRF. `Hit.score` memakai skor kosinus dense (skalanya tetap),
    sehingga ambang `min_score` mengikuti ambang dense.
    """
    kind = 'hybrid'

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25, self.dense, self.rrf_k = bm25, dense, rrf_k
        self.entries = bm25.entries
        self.min_score = dense.min_score

    def search(self, query: str, k: int = 3) -> List[Hit]:
        n = len(self.entries)
        fused: dict = {}
        for r in (self.bm25, self.dense):
            for rank, h in enumerate(r.search(query, n), start=1):
                fused[h.entry.id] = fused.get(h.entry.id, 0.0) + 1.0 / (self.rrf_k + rank)
        cos = {h.entry.id: h.score for h in self.dense.search(query, n)}
        by_id = {e.id: e for e in self.entries}
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(by_id[i], cos[i]) for i in top]
