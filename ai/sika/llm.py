"""Pembungkus Gemini API (SDK google-genai) dengan penanganan error yang rapi."""
from __future__ import annotations

from typing import Optional, Protocol

from . import config


class LLMError(Exception):
    """Dasar semua kegagalan LLM. `code` dipetakan ke kode HTTP di API (lihat README)."""
    code = 503


class LLMTimeout(LLMError):
    code = 504


class LLMQuotaExceeded(LLMError):
    code = 503


class LLMUnavailable(LLMError):
    code = 503


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class GeminiClient:
    """Memanggil Gemini lewat SDK `google-genai`. Kunci API dan nama model dibaca dari .env."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout_s: Optional[float] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL
        self.timeout_s = config.LLM_TIMEOUT_S if timeout_s is None else timeout_s
        if not self.api_key:
            raise ValueError('GEMINI_API_KEY belum diisi. Isi di file .env (jangan di kode).')
        if not self.model:
            raise ValueError('GEMINI_MODEL belum diisi. Pilih model Flash yang tersedia di Google AI Studio '
                             'lalu isi di file .env.')
        from google import genai
        from google.genai import types
        self._types = types
        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),  # dalam milidetik
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import errors
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=config.LLM_TEMPERATURE,
                    max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
                ),
            )
        except errors.APIError as e:
            code = getattr(e, 'code', None)
            if code == 429:
                raise LLMQuotaExceeded('Kuota atau batas permintaan Gemini tercapai.') from e
            if code == 408 or code == 504:
                raise LLMTimeout('Gemini terlalu lama merespons.') from e
            raise LLMUnavailable(f'Layanan Gemini gagal (kode {code}).') from e
        except Exception as e:  # jaringan putus, timeout httpx, dan sebagainya
            name = type(e).__name__.lower()
            if 'timeout' in name:
                raise LLMTimeout('Gemini terlalu lama merespons.') from e
            raise LLMUnavailable('Tidak bisa terhubung ke layanan Gemini.') from e

        text = (getattr(resp, 'text', None) or '').strip()
        if not text:  # misalnya diblokir filter keamanan atau respons kosong
            raise LLMUnavailable('Gemini mengembalikan respons kosong.')
        return text
