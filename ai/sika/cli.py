"""Coba chatbot SIKA lewat terminal.

    python -m sika.cli --dry-run     # tanpa Gemini: memeriksa intent, retrieval, dan prompt
    python -m sika.cli               # dengan Gemini (butuh GEMINI_API_KEY dan GEMINI_MODEL di .env)
"""
from __future__ import annotations

import argparse
import json

from . import config
from .chat import ChatService
from .intent import IntentRouter
from .rag import RagAnswerer, build_retriever


class DryRunLLM:
    """Pengganti Gemini untuk uji alur: menampilkan konteks yang akan dikirim, tanpa memanggil API."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return '[dry-run] Gemini tidak dipanggil. Prompt yang akan dikirim:\n' + user_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description='Coba chatbot SIKA di terminal')
    ap.add_argument('--dry-run', action='store_true', help='jangan panggil Gemini')
    ap.add_argument('--json', action='store_true', help='tampilkan respons lengkap (JSON)')
    args = ap.parse_args()

    if args.dry_run:
        llm = DryRunLLM()
    else:
        from .llm import GeminiClient
        llm = GeminiClient()

    service = ChatService(IntentRouter(), RagAnswerer(build_retriever(), llm))
    print(f'SIKA siap (retriever: {config.RETRIEVER}). Ketik pertanyaan, atau "keluar" untuk berhenti.\n')
    history = []
    while True:
        try:
            q = input('Kamu : ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {'keluar', 'exit', 'quit'}:
            break
        out = service.chat(q, history)
        print(f"SIKA : {out['reply']}\n")
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2), '\n')
        else:
            print(f"       [intent={out['intent']} conf={out['confidence']} action={out['action']} "
                  f"sumber={out['sources']}]\n")
        if out['reply'] and q:
            history += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': out['reply']}]


if __name__ == '__main__':
    main()
