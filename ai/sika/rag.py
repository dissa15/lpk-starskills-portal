"""RAG: retrieval dari Knowledge Base, lalu Gemini menyusun jawaban."""
from __future__ import annotations

from typing import List, Optional

from . import config
from .knowledge import load_kb
from .llm import LLMClient, LLMError
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever


def build_retriever(kind: Optional[str] = None, entries=None):
    """Membuat retriever sesuai pengaturan (`SIKA_RETRIEVER`): bm25, dense, atau hybrid."""
    kind = (kind or config.RETRIEVER).lower()
    if kind not in ('bm25', 'dense', 'hybrid'):
        raise ValueError(f'SIKA_RETRIEVER tidak dikenal: {kind} (pilih bm25, dense, atau hybrid)')
    entries = entries or load_kb()
    if kind == 'bm25':
        return BM25Retriever(entries, min_score=config.MIN_SCORE_BM25)
    dense = DenseRetriever(entries, min_score=config.MIN_SCORE_DENSE,
                           model_name=config.EMBEDDING_MODEL, persist_dir=config.CHROMA_DIR)
    if kind == 'dense':
        return dense
    return HybridRetriever(BM25Retriever(entries, min_score=config.MIN_SCORE_BM25), dense)


class RagAnswerer:
    def __init__(self, retriever, llm: LLMClient, top_k: Optional[int] = None):
        self.retriever, self.llm = retriever, llm
        self.top_k = top_k or config.TOP_K

    def answer(self, question: str, history: Optional[List[dict]] = None) -> dict:
        """Mengembalikan dict:
        reply (str atau None), sources (list id KB), needs_confirmation (bool),
        handoff (bool: True berarti alihkan ke admin), error (None atau dict {code, message}).
        """
        hits = self.retriever.search(question, self.top_k)
        best = hits[0].score if hits else 0.0

        # Gerbang relevansi: info tidak ada di KB, alihkan ke admin tanpa memanggil Gemini.
        if not hits or best < self.retriever.min_score:
            return {'reply': None, 'sources': [], 'needs_confirmation': False,
                    'handoff': True, 'error': None}

        entries = [h.entry for h in hits]
        recent = (history or [])[-2 * config.HISTORY_TURNS:] if config.HISTORY_TURNS > 0 else None
        try:
            reply = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, entries, recent))
        except LLMError as e:
            return {'reply': None, 'sources': [h.entry.id for h in hits], 'needs_confirmation': False,
                    'handoff': True, 'error': {'code': e.code, 'message': str(e)}}

        return {'reply': reply, 'sources': [e.id for e in entries],
                'needs_confirmation': any(e.needs_confirmation for e in entries),
                'handoff': False, 'error': None}
