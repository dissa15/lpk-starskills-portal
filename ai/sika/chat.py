"""Orkestrasi satu giliran chat: klasifikasi intent, lalu RAG atau balasan statis."""
from __future__ import annotations

from typing import List, Optional

from .intent import IntentRouter
from .rag import RagAnswerer
from .responses import static_reply, whatsapp_url

WHATSAPP_ACTIONS = {'handoff_admin', 'show_contact_whatsapp'}


class ChatService:
    def __init__(self, router: IntentRouter, rag: RagAnswerer):
        self.router, self.rag = router, rag

    def chat(self, message: str, history: Optional[List[dict]] = None) -> dict:
        """Respons siap kirim ke Back-End:
        reply, intent, confidence, action, handoff, whatsapp_url, sources, needs_confirmation, error.
        """
        r = self.router.route(message)
        action = r['action']
        out = {'reply': None, 'intent': r['intent'], 'confidence': r['confidence'], 'action': action,
               'handoff': False, 'whatsapp_url': None, 'sources': [],
               'needs_confirmation': False, 'error': None}

        if action == 'answer_with_rag':
            a = self.rag.answer(message, history)
            out.update(sources=a['sources'], needs_confirmation=a['needs_confirmation'], error=a['error'])
            if a['handoff']:
                out.update(action='handoff_admin', handoff=True, reply=static_reply('handoff_admin'),
                           whatsapp_url=whatsapp_url())
            else:
                out['reply'] = a['reply']
            return out

        out['reply'] = static_reply(action)
        if action in WHATSAPP_ACTIONS:
            out['whatsapp_url'] = whatsapp_url()
        out['handoff'] = action == 'handoff_admin'
        return out
