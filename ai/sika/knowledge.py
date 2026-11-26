"""Pemuatan Knowledge Base (KB) dari CSV."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import config


@dataclass(frozen=True)
class Entry:
    id: str
    kategori: str
    judul: str
    isi: str
    status: str  # 'terverifikasi' atau 'perlu_konfirmasi'

    @property
    def text(self) -> str:
        """Teks gabungan yang dipakai untuk pencarian."""
        return f'{self.judul}. {self.isi}'

    @property
    def needs_confirmation(self) -> bool:
        return self.status.strip().lower() != 'terverifikasi'


def load_kb(path: Optional[Path] = None) -> List[Entry]:
    path = Path(path or config.KB_PATH)
    if not path.exists():
        raise FileNotFoundError(f'Knowledge Base tidak ditemukan di {path}')
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    entries = [Entry(r['id'], r['kategori'], r['judul'], r['isi'], r['status_verifikasi']) for r in rows]
    if not entries:
        raise ValueError(f'Knowledge Base kosong: {path}')
    return entries
