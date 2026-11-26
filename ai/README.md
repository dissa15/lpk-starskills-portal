# Modul AI SIKA (tim Gen AI)

Intent classifier, RAG (retrieval + Gemini), dan orkestrasi chat untuk chatbot SIKA.
Siap di-import oleh FastAPI.

## Alur satu pesan
```
pesan -> IntentRouter (model buatan sendiri)
  - di_luar_konteks            -> tolak sopan            (tanpa Gemini)
  - minta_rekomendasi_jurusan  -> arahkan ke kuis        (tanpa Gemini)
  - tanya_kontak_lokasi        -> kontak + WhatsApp      (tanpa Gemini)
  - sapaan_penutup             -> sapaan SIKA            (tanpa Gemini)
  - confidence rendah          -> alihkan ke admin       (tanpa Gemini)
  - lainnya                    -> RAG:
        cari di Knowledge Base -> skor di bawah ambang?  alihkan ke admin (tanpa Gemini)
                               -> cukup relevan?         Gemini menyusun jawaban dari entri KB
        Gemini gagal / timeout / kuota -> alihkan ke admin + kode error
```
Gemini hanya dipanggil bila perlu, jadi kuota gratis lebih awet.

## Isi
| File | Fungsi |
|---|---|
| `sika/preprocess.py` | Preprocessing teks (harus sama dengan saat melatih model) |
| `sika/train.py` | Melatih ulang intent classifier dari `data/intent_dataset.csv` |
| `sika/intent.py` | `IntentRouter`: intent, confidence, aksi |
| `sika/knowledge.py` | Memuat `data/knowledge_base.csv` |
| `sika/retrieval.py` | BM25 (bawaan), dense (embedding + ChromaDB), hybrid (RRF) |
| `sika/prompts.py` | Prompt sistem SIKA dan format konteks |
| `sika/llm.py` | Pembungkus Gemini (`google-genai`) dengan penanganan error |
| `sika/rag.py` | `RagAnswerer`: gerbang relevansi + Gemini |
| `sika/chat.py` | `ChatService`: satu giliran chat lengkap |
| `sika/responses.py` | Balasan statis (sapaan, tolak sopan, kuis, kontak, alih ke admin) |
| `sika/config.py` | Semua pengaturan (bisa diganti lewat `.env`) |
| `sika/cli.py` | Coba chatbot di terminal |
| `tests/` | 30 uji otomatis (tanpa memanggil Gemini asli) |

## Mulai
```bash
cd ai
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # cukup untuk BM25 + Gemini (bawaan)
# pip install -r requirements-dense.txt   # hanya jika memakai dense/hybrid (ChromaDB + embedding)
python -m sika.train               # membuat models/intent_classifier.joblib
pytest -q                          # semua uji harus lulus (uji dense dilewati bila chromadb belum terpasang)
python -m sika.cli --dry-run       # coba alur tanpa Gemini
```

Mengaktifkan Gemini:
1. Buat API key di Google AI Studio.
2. Salin `.env.example` menjadi `.env`, isi `GEMINI_API_KEY` dan `GEMINI_MODEL`
   (pilih model Flash yang tersedia di akunmu, nama model sering berubah).
3. `python -m sika.cli`

`.env` sudah ada di `.gitignore`. **Jangan pernah commit key.** Kalau sempat ter-commit, buat key baru.

Latih model di perangkat masing-masing (jangan memakai `.joblib` dari perangkat lain),
lalu `pip freeze > requirements.lock.txt` supaya versi library sama satu tim.

## Contoh FastAPI
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sika import ChatService, IntentRouter
from sika.llm import GeminiClient
from sika.rag import RagAnswerer, build_retriever

app = FastAPI(title="SIKA AI Service")
service = ChatService(IntentRouter(), RagAnswerer(build_retriever(), GeminiClient()))

class ChatIn(BaseModel):
    message: str = Field(min_length=1)
    history: list[dict] = []      # [{"role": "user"|"assistant", "content": "..."}]

@app.post("/chat")
def chat(body: ChatIn):
    out = service.chat(body.message, body.history)
    if out["action"] == "invalid_input":
        raise HTTPException(400, out["reply"])
    return out                    # out["error"] berisi {code, message} jika Gemini gagal
```
Bentuk respons `chat()`: `reply, intent, confidence, action, handoff, whatsapp_url, sources,
needs_confirmation, error`. Saat Gemini gagal, `reply` tetap berisi pesan ramah + tautan WhatsApp
dan `error.code` bernilai 503 (kuota/layanan gagal) atau 504 (timeout), sehingga web tidak perlu crash.

## Catatan kalibrasi dan hal yang belum terverifikasi
- **Ambang BM25 (1,5)** dikalibrasi kasar dari 26 query evaluasi dan 10 query "tidak ada di KB":
  sekitar 94% query langsung tetap dijawab, 60% query tanpa jawaban di KB dialihkan.
  Query parafrase hanya 25% lolos karena BM25 mengandalkan kata yang sama. Itu alasan
  dense/hybrid disediakan. Pilih `SIKA_RETRIEVER` dari tabel di notebook retrieval kalian.
- **Ambang dense (0,35)** hanyalah asumsi awal. Ganti dengan nilai dari tabel ambang di notebook retrieval (bagian 7).
- Panggilan ke Gemini yang sesungguhnya belum diuji di lingkungan pembuatan modul ini (tanpa akses jaringan
  dan tanpa API key). Yang sudah diuji: seluruh alur dengan LLM palsu, bentuk pemanggilan SDK, dan
  pemetaan error. Uji dengan key aslimu sebelum diintegrasikan.
- Prompt SIKA (`sika/prompts.py`) belum diuji pada pertanyaan nyata. Uji dengan pertanyaan di luar konteks
  (minimal 10 skenario, syarat Side Quest) dan pertanyaan yang jawabannya tidak ada di KB.
- Rute kuis (`SIKA_QUIZ_URL`) belum disepakati dengan tim Full Stack.
