"""Uji dasar modul intent. Jalankan dari folder ai/:  pytest -q"""
import pytest

from sika.intent import ALL_ACTIONS, IntentRouter
from sika.preprocess import preprocess
from sika.responses import static_reply, whatsapp_url
from sika import config


def test_preprocess():
    assert preprocess('  Halo, Makasih!! ') == 'halo terima kasih'
    assert preprocess('Alamat lpk dimana?') == 'alamat lpk di mana'
    assert preprocess('') == ''


def test_route_structure(router):
    out = router.route('Syarat daftarnya apa saja?')
    assert set(out) == {'intent', 'confidence', 'action'}
    assert out['action'] in ALL_ACTIONS
    assert 0.0 <= out['confidence'] <= 1.0


def test_empty_input_is_invalid(router):
    assert router.route('')['action'] == 'invalid_input'
    assert router.route('   ')['action'] == 'invalid_input'
    assert router.route('???')['action'] == 'invalid_input'


@pytest.mark.parametrize('text,action', [
    ('Syarat daftarnya apa saja?', 'answer_with_rag'),
    ('Cocoknya aku ambil jurusan apa?', 'redirect_quiz'),
    ('Alamat lpk dimana', 'show_contact_whatsapp'),
    ('makasih kak', 'greet'),
])
def test_known_examples(router, text, action):
    assert router.route(text)['action'] == action


def test_low_confidence_goes_to_admin(router):
    strict = IntentRouter(model_path=router.model_path, threshold=0.99)
    assert strict.route('Syarat daftarnya apa saja?')['action'] == 'handoff_admin'


def test_static_replies():
    for action in ALL_ACTIONS:
        if action == 'answer_with_rag':
            assert static_reply(action) is None
        else:
            assert static_reply(action)
    assert whatsapp_url().startswith('https://wa.me/')


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        IntentRouter(model_path=tmp_path / 'tidak_ada.joblib')
