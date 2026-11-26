"""Balasan statis untuk aksi yang tidak memerlukan RAG/Gemini."""
from __future__ import annotations

from typing import Optional
from urllib.parse import quote

from . import config

GREETING = (
    f'Halo! Aku {config.BOT_NAME} 👋\n'
    'Asisten virtual Star Skills Innovations. Aku siap membantu kamu mendapatkan '
    'informasi seputar program pelatihan, persyaratan, dan proses pendaftaran di LPK.'
)


def whatsapp_url(draft: Optional[str] = None) -> str:
    """Tautan WhatsApp ke admin LPK dengan draf pesan otomatis."""
    text = quote(draft or config.WHATSAPP_DRAFT)
    return f'https://wa.me/{config.WHATSAPP_NUMBER}?text={text}'


def static_reply(action: str) -> Optional[str]:
    """Teks balasan untuk aksi non-RAG. Mengembalikan None untuk 'answer_with_rag'
    (jawabannya dibuat oleh RAG + Gemini)."""
    wa = whatsapp_url()
    replies = {
        'greet': GREETING,
        'refuse_politely': (
            'Maaf, aku hanya bisa membantu informasi seputar LPK Star Skills Innovations, '
            'seperti program pelatihan, persyaratan, dan pendaftaran. '
            'Ada yang ingin kamu tanyakan soal itu?'
        ),
        'redirect_quiz': (
            'Untuk menemukan jurusan yang paling cocok, coba kuis rekomendasi jurusan kami '
            f'di sini: {config.QUIZ_URL}'
        ),
        'show_contact_whatsapp': (
            f'Kamu bisa menghubungi admin LPK lewat WhatsApp: {wa}\n'
            f'Alamat: {config.ADDRESS}\n'
            f'Instagram: {config.INSTAGRAM}'
        ),
        'handoff_admin': (
            'Aku belum punya informasi yang pasti untuk pertanyaan itu, jadi supaya tidak keliru '
            f'kamu bisa tanya langsung ke admin LPK lewat WhatsApp: {wa}'
        ),
        'invalid_input': (
            'Pesanmu masih kosong. Silakan ketik pertanyaan tentang program pelatihan, '
            'persyaratan, atau pendaftaran di LPK.'
        ),
    }
    return replies.get(action)
