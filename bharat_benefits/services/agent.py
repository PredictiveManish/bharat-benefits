"""
Agent service - handles LLM interaction with Sarvam AI.
Manages conversation history and generates grounded answers.
"""

import json
import os
import re
from typing import Optional

import httpx

from .knowledge_base import KnowledgeBase

# -- Config -------------------------------------------------------------------

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE = "https://api.sarvam.ai"
CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b")
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "1500"))

SYSTEM_PROMPT = """You are a voice assistant that answers questions about Indian
government welfare schemes using only the retrieved context provided.

Rules:
- Answer only from the retrieved context. If the context is insufficient, say so and direct the user to the official government portal.
- If you have strong prior knowledge about a scheme mentioned in the context, you may add it — but never invent eligibility rules, amounts, deadlines, or deadlines that are not in the context or your known facts.
- Respond in the SAME LANGUAGE the user asked their question. Detect the language automatically.
- Keep answers short and suitable for listening — no bullet points, no tables, no markdown, no numbered lists.
- 3 to 5 sentences maximum.
- Mention the helpline number if one is present in the context.
- Always end with: "Please verify the details on the official government portal." (in the same language as the response)."""


class Agent:
    """Voice agent that combines knowledge base retrieval with LLM answer generation."""

    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.conversation_history = []
        self.max_history_messages = 6  # last 3 turns

    def retrieve(self, question: str) -> tuple[str, list[dict]]:
        """Retrieve relevant context from the knowledge base."""
        results = self.knowledge_base.query(question)
        context = self.knowledge_base.get_context_string(results)
        return context, results

    def generate_answer(
        self,
        question: str,
        context: str,
        results: Optional[list[dict]] = None,
    ) -> str:
        """Generate a voice-friendly answer using the LLM."""
        user_msg = (
            f"User question: {question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Give a short, voice-friendly answer in 3 to 5 sentences."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.conversation_history[-self.max_history_messages:])
        messages.append({"role": "user", "content": user_msg})

        print("   generating answer ...")

        resp = httpx.post(
            f"{SARVAM_BASE}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {SARVAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.2,
                "reasoning_effort": None,
            },
            timeout=60.0,
        )
        resp.raise_for_status()

        try:
            json_resp = resp.json()
        except Exception:
            raise ValueError(f"LLM returned non-JSON response: {resp.text[:200]}")
        
        choices = json_resp.get("choices", [])
        if not choices:
            error_info = json_resp.get("error", {})
            raise ValueError(f"Empty response from LLM. Error: {error_info.get('message', 'Unknown')} (code: {error_info.get('code', 'N/A')})")
        
        raw = choices[0]["message"]["content"]
        if raw is None:
            raise ValueError(f"LLM returned None content. Full response: {json.dumps(json_resp, indent=2)[:500]}")
        raw = raw.strip()
        answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        preview = answer[:120] + ("..." if len(answer) > 120 else "")
        print(f"   answer: {preview}")

        return answer

    def process_turn(self, question: str) -> str:
        """Process a full turn: retrieve context, generate answer, update history."""
        context, results = self.retrieve(question)
        answer = self.generate_answer(question, context, results)

        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return answer

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
