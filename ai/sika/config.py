"""Pengaturan terpusat. Nilai bisa diganti lewat environment variable (.env), tanpa mengubah kode."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

try:  # baca file .env (jika ada) sebelum nilai di bawah dibaca
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / '.env')
except ImportError:
    pass

MODEL_PATH = Path(os.getenv('SIKA_MODEL_PATH', BASE_DIR / 'models' / 'intent_classifier.joblib'))
DATA_PATH = Path(os.getenv('SIKA_DATA_PATH', BASE_DIR / 'data' / 'intent_dataset.csv'))
KB_PATH = Path(os.getenv('SIKA_KB_PATH', BASE_DIR / 'data' / 'knowledge_base.csv'))

# Ambang confidence: di bawah ini pertanyaan dialihkan ke admin.
# Nilai 0.25 dipilih dari tabel di notebook intent (bagian 7). Sesuaikan jika data bertambah.
CONF_THRESHOLD = float(os.getenv('SIKA_CONF_THRESHOLD', '0.25'))

# Kontak resmi LPK (dari profil LPK).
WHATSAPP_NUMBER = os.getenv('SIKA_WHATSAPP_NUMBER', '6281907035915')   # format internasional tanpa +
WHATSAPP_DRAFT = os.getenv('SIKA_WHATSAPP_DRAFT', 'Halo admin LPK Star Skills, saya ingin bertanya lebih lanjut.')
ADDRESS = ('Jl. Kusuma No.7, RT.01/RW.04, Berbek, Kecamatan Waru, '
           'Kabupaten Sidoarjo, Jawa Timur 61256')
INSTAGRAM = 'instagram.com/starskillsinnovation'

# Rute kuis rekomendasi jurusan di web. ASUMSI: belum disepakati dengan tim Full Stack, sesuaikan.
QUIZ_URL = os.getenv('SIKA_QUIZ_URL', '/rekomendasi-jurusan')

BOT_NAME = 'SIKA'


# ---- RAG ----
# Metode retrieval: bm25 (default, ringan), dense (embedding + ChromaDB), atau hybrid.
# Pilih berdasarkan tabel hasil di notebook retrieval, lalu atur lewat .env.
RETRIEVER = os.getenv('SIKA_RETRIEVER', 'bm25').lower()
TOP_K = int(os.getenv('SIKA_TOP_K', '3'))

# Ambang skor teratas: di bawahnya dianggap "informasi tidak ada di KB" lalu dialihkan ke admin
# TANPA memanggil Gemini (menghemat kuota).
# BM25 1.5: dikalibrasi kasar dari 26 query evaluasi + 10 query "tidak ada di KB" (lihat README).
# Dense 0.35: ASUMSI awal, WAJIB dikalibrasi ulang dengan tabel ambang di notebook retrieval (bagian 7).
MIN_SCORE_BM25 = float(os.getenv('SIKA_MIN_SCORE_BM25', '1.5'))
MIN_SCORE_DENSE = float(os.getenv('SIKA_MIN_SCORE_DENSE', '0.35'))
EMBEDDING_MODEL = os.getenv('SIKA_EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
CHROMA_DIR = os.getenv('SIKA_CHROMA_DIR') or None   # kosong = ChromaDB di memori

# ---- Gemini ----
# Nama model Gemini berubah cepat. WAJIB diisi di .env dengan model Flash yang tersedia untuk akunmu
# (cek daftar model di Google AI Studio). Sengaja tanpa nilai bawaan.
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', '')
LLM_TIMEOUT_S = float(os.getenv('SIKA_LLM_TIMEOUT_S', '10'))   # sama dengan batas fallback di project plan
LLM_TEMPERATURE = float(os.getenv('SIKA_LLM_TEMPERATURE', '0.2'))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv('SIKA_LLM_MAX_OUTPUT_TOKENS', '512'))
HISTORY_TURNS = int(os.getenv('SIKA_HISTORY_TURNS', '4'))       # jumlah giliran chat sebelumnya yang dikirim
