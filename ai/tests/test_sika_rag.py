"""Uji RAG dan ChatService tanpa memanggil Gemini asli (memakai LLM palsu)."""
import hashlib

import numpy as np
import pytest

from sika import config
from sika.chat import ChatService
from sika.intent import IntentRouter
from sika.knowledge import load_kb
from sika.llm import GeminiClient, LLMQuotaExceeded, LLMTimeout, LLMUnavailable
from sika.prompts import SYSTEM_PROMPT, build_user_prompt
from sika.rag import RagAnswerer, build_retriever
from sika.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


class FakeLLM:
    def __init__(self, reply='Jawaban uji dari SIKA.', error=None):
        self.reply, self.error, self.calls = reply, error, []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture(scope='module')
def kb():
    return load_kb()


@pytest.fixture()
def bm25(kb):
    return BM25Retriever(kb, min_score=config.MIN_SCORE_BM25)


def hash_embed(texts):
    """Embedding tiruan (hash n-gram karakter) agar uji tidak perlu mengunduh model."""
    out = np.zeros((len(texts), 256))
    for i, t in enumerate(texts):
        t = ' ' + t.lower() + ' '
        for j in range(len(t) - 2):
            out[i, int(hashlib.md5(t[j:j + 3].encode()).hexdigest(), 16) % 256] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1
    return (out / n).tolist()


# ---------- Knowledge Base dan retrieval ----------
def test_kb_loaded(kb):
    assert len(kb) == 17
    assert {e.id for e in kb} >= {'KB01', 'KB17'}
    assert any(e.needs_confirmation for e in kb)


def test_bm25_finds_relevant_entries(bm25):
    ids = [h.entry.id for h in bm25.search('syarat umur dan pendidikan untuk daftar', 3)]
    assert 'KB06' in ids
    assert bm25.search('alamat lembaganya di mana', 1)[0].entry.id == 'KB15'


def test_dense_with_chroma_and_hybrid(kb, bm25):
    pytest.importorskip('chromadb')   # opsional: hanya untuk dense/hybrid
    dense = DenseRetriever(kb, min_score=0.0, embed_fn=hash_embed, collection_name='test_kb')
    hits = dense.search('nomor whatsapp admin', 3)
    assert len(hits) == 3 and hits[0].score >= hits[-1].score
    hybrid = HybridRetriever(bm25, dense)
    assert len(hybrid.search('nomor whatsapp admin', 3)) == 3


def test_build_retriever_rejects_unknown_kind(kb):
    with pytest.raises(ValueError):
        build_retriever('tidak_ada', kb)


# ---------- RagAnswerer ----------
def test_rag_answers_with_context(bm25):
    llm = FakeLLM()
    out = RagAnswerer(bm25, llm).answer('Apa saja syarat peserta pelatihan?')
    assert out['reply'] == 'Jawaban uji dari SIKA.' and not out['handoff'] and out['error'] is None
    assert 'KB06' in out['sources']
    system_prompt, user_prompt = llm.calls[0]
    assert system_prompt == SYSTEM_PROMPT
    assert 'usia minimal 19 tahun' in user_prompt          # isi KB ikut dikirim sebagai konteks
    assert 'PERTANYAAN PENGGUNA' in user_prompt


def test_rag_gate_hands_off_without_calling_llm(bm25):
    llm = FakeLLM()
    out = RagAnswerer(bm25, llm).answer('Mau tanya soal asrama')
    assert out['handoff'] and out['reply'] is None
    assert llm.calls == []                                   # hemat kuota Gemini


def test_rag_flags_unverified_info(bm25):
    out = RagAnswerer(bm25, FakeLLM()).answer('Berapa lama durasi pelatihan motor injeksi?')
    assert 'KB04' in out['sources'] and out['needs_confirmation']
    # entri belum terverifikasi diberi tanda di prompt
    entries = [e for e in load_kb() if e.id == 'KB04']
    assert '[perlu_konfirmasi]' in build_user_prompt('tes', entries)


@pytest.mark.parametrize('exc,code', [
    (LLMTimeout('lambat'), 504), (LLMQuotaExceeded('kuota'), 503), (LLMUnavailable('mati'), 503)])
def test_rag_llm_errors_become_handoff(bm25, exc, code):
    out = RagAnswerer(bm25, FakeLLM(error=exc)).answer('Apa saja syarat peserta pelatihan?')
    assert out['handoff'] and out['reply'] is None and out['error']['code'] == code


def test_history_is_truncated(bm25):
    llm = FakeLLM()
    history = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'pesan-{i}'} for i in range(20)]
    RagAnswerer(bm25, llm).answer('Apa saja syarat peserta pelatihan?', history)
    prompt = llm.calls[0][1]
    assert 'pesan-19' in prompt and 'pesan-0' not in prompt


# ---------- ChatService ----------
@pytest.fixture()
def service(router, bm25):
    llm = FakeLLM()
    return ChatService(router, RagAnswerer(bm25, llm)), llm


def test_chat_greeting_skips_llm(service):
    svc, llm = service
    out = svc.chat('makasih kak')
    assert out['action'] == 'greet' and 'SIKA' in out['reply'] and llm.calls == []


def test_chat_rag_path(service):
    svc, llm = service
    out = svc.chat('Syarat daftarnya apa saja?')
    assert out['action'] == 'answer_with_rag' and out['reply'] == 'Jawaban uji dari SIKA.'
    assert out['sources'] and len(llm.calls) == 1 and not out['handoff']


def test_chat_out_of_context_and_quiz_skip_llm(service):
    svc, llm = service
    assert svc.chat('Siapa presiden Indonesia?')['action'] == 'refuse_politely'
    quiz = svc.chat('Cocoknya aku ambil jurusan apa?')
    assert quiz['action'] == 'redirect_quiz' and config.QUIZ_URL in quiz['reply']
    assert llm.calls == []


def test_chat_contact_has_whatsapp(service):
    out = service[0].chat('Alamat lpk dimana')
    assert out['action'] == 'show_contact_whatsapp' and out['whatsapp_url'].startswith('https://wa.me/')


def test_chat_low_classifier_confidence_goes_to_admin(router, bm25):
    llm = FakeLLM()
    strict = IntentRouter(model_path=router.model_path, threshold=0.99)
    out = ChatService(strict, RagAnswerer(bm25, llm)).chat('Syarat daftarnya apa saja?')
    assert out['action'] == 'handoff_admin' and out['handoff'] and out['whatsapp_url'] and llm.calls == []


def test_chat_rag_gate_handoff_sets_whatsapp(router, bm25):
    llm = FakeLLM()
    svc = ChatService(router, RagAnswerer(bm25, llm))
    out = svc.chat('Apakah ada kelas akhir pekan di sini?')
    if out['intent'] != 'di_luar_konteks':
        assert out['handoff'] and out['whatsapp_url'] and out['reply']


def test_chat_llm_failure_gives_graceful_reply(router, bm25):
    svc = ChatService(router, RagAnswerer(bm25, FakeLLM(error=LLMTimeout('lambat'))))
    out = svc.chat('Syarat daftarnya apa saja?')
    assert out['handoff'] and out['error']['code'] == 504 and 'WhatsApp' in out['reply']


# ---------- GeminiClient (tanpa jaringan) ----------
def test_gemini_requires_key_and_model():
    with pytest.raises(ValueError):
        GeminiClient(api_key='', model='m')
    with pytest.raises(ValueError):
        GeminiClient(api_key='k', model='')


def test_gemini_error_mapping():
    from google.genai import errors
    client = GeminiClient(api_key='kunci-palsu', model='model-palsu')

    def raising(code):
        def f(**kwargs):
            raise errors.ClientError(code, {'error': {'message': 'x', 'status': 'X'}})
        return f

    client._client.models.generate_content = raising(429)
    with pytest.raises(LLMQuotaExceeded):
        client.generate('s', 'u')
    client._client.models.generate_content = raising(400)
    with pytest.raises(LLMUnavailable):
        client.generate('s', 'u')

    class Resp:
        text = '  '
    client._client.models.generate_content = lambda **kw: Resp()
    with pytest.raises(LLMUnavailable):
        client.generate('s', 'u')

    Resp.text = 'ok'
    assert client.generate('s', 'u') == 'ok'
