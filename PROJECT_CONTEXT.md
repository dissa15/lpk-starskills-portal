Terakhir diperbarui: 26 November 2026

# Konteks Proyek: Chatbot SIKA (Tim Gen AI)

Dokumen ini dibaca lebih dulu oleh anggota tim dan AI assistant sebelum mengerjakan bagian AI. Perbarui setiap ada keputusan baru.

## Proyek
Capstone Asah 2026 (tim A26-CS127). UMKM: LPK Star Skills Innovations (lembaga pelatihan kerja di Sidoarjo). Solusi: web profil interaktif + CMS admin + asisten virtual bernama **SIKA**.

Tim: 2 Full Stack (Express, Vite, Tailwind, Axios, PostgreSQL), 2 Data Science (kuis rekomendasi jurusan, dashboard), 2 Gen AI.

## Arsitektur
Front-End (Vite) -> Back-End (Express) -> layanan AI (FastAPI) -> Gemini API dengan RAG.
Back-End menyimpan riwayat chat di database. Front-End tidak boleh memanggil layanan AI atau Gemini langsung.

## Perilaku SIKA
| Situasi | Aksi (`action`) |
|---|---|
| Pertanyaan dalam topik dan ada di Knowledge Base | `answer_with_rag` (Gemini menyusun jawaban dari KB) |
| Di luar konteks LPK | `refuse_politely` |
| Minta rekomendasi jurusan | `redirect_quiz` (arahkan ke kuis tim DS, SIKA tidak merekomendasikan sendiri) |
| Tanya kontak atau lokasi | `show_contact_whatsapp` |
| Sapaan atau penutup | `greet` |
| Tidak ada di KB, confidence rendah, atau Gemini gagal | `handoff_admin` (alihkan ke admin lewat WhatsApp) |
| Pesan kosong | `invalid_input` |

SIKA tidak mendaftarkan peserta dan tidak menjawab informasi yang belum terverifikasi dari sumber resmi LPK.

## Aturan wajib dari Asah (jangan dilanggar)
- Model AI utama harus dibangun sendiri dan dilatih dari data tim. DILARANG: model pretrained (TensorFlow Hub dan sejenisnya) sebagai model utama, memakai Gemini/ChatGPT API sebagai pengganti model buatan sendiri, AutoML, atau fork repositori AI milik orang lain.
- Model pendukung siap pakai (embedding, OCR, LLM) hanya boleh untuk fitur Side Quest.
- Data dummy wajib diberi label jelas beserta cara pembuatannya.
- Bandingkan minimal 2 model atau konfigurasi dengan metrik yang sama, tabelnya masuk README.
- Layanan AI diakses lewat Back-End, model dilayani lewat REST API (FastAPI), dokumentasi dengan Swagger/OpenAPI.

## Status bagian AI
Sudah ada di folder `ai/`:
- Dataset intent (`ai/data/intent_dataset.csv`): 224 kalimat, 8 intent, data dummy buatan tim (kolom `source`).
- Knowledge Base (`ai/data/knowledge_base.csv`): 17 entri, kolom `status_verifikasi` (terverifikasi atau perlu_konfirmasi).
- Intent classifier buatan sendiri: TF-IDF kata + karakter n-gram dengan Logistic Regression. Macro-F1 data uji sekitar 0,74 (data kecil, hasil indikatif). Notebook di `ai/notebooks/`.
- Eksperimen retrieval (TF-IDF, BM25, dense, hybrid) dengan Precision@k, MRR, nDCG. Notebook di `ai/notebooks/`.
- Paket Python `ai/sika/`: preprocessing, training, `IntentRouter`, retrieval, prompt SIKA, pembungkus Gemini, `RagAnswerer`, `ChatService`, CLI, dan 30 uji. Lihat `ai/README.md`.

Belum selesai atau belum diuji:
- Panggilan Gemini asli (butuh API key dan nama model di `.env`).
- Skenario uji penolakan di luar konteks (minimal 10) dan uji error handling.
- Endpoint FastAPI final dan API contract dengan tim Full Stack.
- Pipeline OCR dan pemrosesan gambar.
- Penambahan data intent dari anggota tim lain, lalu latih ulang.
- README akhir bagian AI.

## Cara menjalankan modul AI
```bash
cd ai
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   |   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m sika.train
pytest -q
python -m sika.cli --dry-run
```
Latih model di perangkat masing-masing (file `.joblib` terikat versi scikit-learn). Jangan pernah commit `.env` atau API key.

## Cara kerja bersama
- Satu orang satu branch (misalnya `ai-nama`), jangan push langsung ke `main`. Gabungkan lewat pull request.
- Bagian AI ada di folder `ai/`. Jangan mengubah file orang lain tanpa kabar.
- Jangan memakai `git add .`. Tambahkan file dengan menyebut namanya.
- Jangan commit `.env`, `.venv`, `node_modules`, atau file model `.joblib`.
- Jangan memakai tanda pisah panjang (em dash atau en dash) di teks, README, atau komentar.
- Jelaskan kode dengan komentar singkat dalam bahasa Indonesia dan sebutkan asumsi yang dibuat.
