"""Klasifikasi intent dan routing aksi untuk chatbot SIKA."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib

from . import config
from .preprocess import preprocess

# intent -> aksi. Intent yang tidak terdaftar di sini otomatis menjadi 'answer_with_rag'.
INTENT_ACTIONS = {
    'di_luar_konteks': 'refuse_politely',
    'minta_rekomendasi_jurusan': 'redirect_quiz',
    'tanya_kontak_lokasi': 'show_contact_whatsapp',
    'sapaan_penutup': 'greet',
}
DEFAULT_ACTION = 'answer_with_rag'

# Semua nilai action yang mungkin dikembalikan (untuk kontrak API dan validasi).
ALL_ACTIONS = ('answer_with_rag', 'refuse_politely', 'redirect_quiz',
               'show_contact_whatsapp', 'greet', 'handoff_admin', 'invalid_input')


class IntentRouter:
    """Memuat model sekali, lalu dipakai berulang kali (cocok dipanggil dari FastAPI)."""

    def __init__(self, model_path: Optional[Path] = None, threshold: Optional[float] = None):
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.threshold = config.CONF_THRESHOLD if threshold is None else float(threshold)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f'Model tidak ditemukan di {self.model_path}. '
                'Latih dulu dengan: python -m sika.train'
            )
        self.model = joblib.load(self.model_path)

    def route(self, question: str) -> dict:
        """Mengembalikan {'intent', 'confidence', 'action'}."""
        q = preprocess(question)
        if not q:
            return {'intent': None, 'confidence': 0.0, 'action': 'invalid_input'}

        proba = self.model.predict_proba([q])[0]
        idx = int(proba.argmax())
        intent, conf = str(self.model.classes_[idx]), float(proba[idx])

        if conf < self.threshold:
            action = 'handoff_admin'
        else:
            action = INTENT_ACTIONS.get(intent, DEFAULT_ACTION)
        return {'intent': intent, 'confidence': round(conf, 3), 'action': action}


_default_router: Optional[IntentRouter] = None


def route(question: str) -> dict:
    """Jalan pintas: memakai satu IntentRouter bersama (dimuat saat pertama kali dipanggil)."""
    global _default_router
    if _default_router is None:
        _default_router = IntentRouter()
    return _default_router.route(question)
