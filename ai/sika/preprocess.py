"""Preprocessing teks. HARUS sama persis dengan yang dipakai saat melatih model.

Kalau fungsi ini diubah, latih ulang model (python -m sika.train),
karena model lama dilatih dengan versi preprocessing sebelumnya.
"""
import re

SLANG = {
    'nggak': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'tdk': 'tidak',
    'aja': 'saja', 'yg': 'yang', 'dgn': 'dengan',
    'dmn': 'dimana', 'dimana': 'di mana', 'kpn': 'kapan', 'brp': 'berapa',
    'wa': 'whatsapp', 'min': 'admin',
    'thanks': 'terima kasih', 'makasih': 'terima kasih',
}


def preprocess(text) -> str:
    """Huruf kecil, hapus tanda baca, normalisasi singkatan, rapikan spasi."""
    t = str(text).lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = ' '.join(SLANG.get(w, w) for w in t.split())
    return re.sub(r'\s+', ' ', t).strip()
