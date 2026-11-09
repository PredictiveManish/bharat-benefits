"""
main.py - Bharat Benefits Voice Agent
======================================
Continuous voice loop:

    mic -> Sarvam STT -> ChromaDB RAG -> Sarvam LLM -> Sarvam TTS -> speaker
           (repeats until you say "exit" or press Ctrl+C)

Two recording modes:
  - Fixed window (default): records for a set number of seconds per turn.
  - VAD mode (--vad): detects silence automatically.

Usage:
    python main.py                  # fixed 5s window
    python main.py --seconds 8      # fixed 8s window
    python main.py --vad            # voice activity detection
    python main.py --text-mode      # type instead of speak (debug)

Environment (.env):
    SARVAM_API_KEY         - Sarvam subscription key
    SARVAM_STT_MODEL       - default: saaras:v3
    SARVAM_CHAT_MODEL      - default: sarvam-105b
    SARVAM_TTS_MODEL       - default: bulbul:v3
    SARVAM_TTS_SPEAKER     - default: priya
    CHROMA_DB_PATH         - ChromaDB storage path (default: ./chroma_data)
    CHROMA_COLLECTION      - Collection name (default: bharat-benefits)
    EMBEDDING_MODEL        - Sentence transformer (default: all-MiniLM-L6-v2)
    TOP_K                  - Results per query (default: 3)

Install:
    pip install -r requirements.txt
    python ingest.py              # build knowledge base first
    python main.py                # start the agent
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment
load_dotenv()

from bharat_benefits.services.audio import (
    record_fixed,
    record_vad,
    transcribe,
    speak,
)
from bharat_benefits.services.agent import Agent
from bharat_benefits.services.knowledge_base import KnowledgeBase

# -- Config -------------------------------------------------------------------

SAMPLE_RATE = 16_000

EXIT_PHRASES = {
    "exit",
    "quit",
    "stop",
    "bye",
    "goodbye",
    "band karo",
    "band kar",
    "बंद करो",
}


async def live_loop(
    record_seconds: int = 5, use_vad: bool = False, text_mode: bool = False
):
    """Continuous voice-to-voice conversation loop."""
    # Initialize knowledge base and agent
    print("\nLoading knowledge base ...")
    kb = KnowledgeBase()
    agent = Agent(knowledge_base=kb)

    # Report status
    info = kb.get_or_create_collection()
    print(f"Ready: {info['name']} ({info['document_count']} chunks)")

    turn = 0

    mode_label = "vad" if use_vad else f"{record_seconds}s fixed"
    if text_mode:
        mode_label = "text"

    print("-" * 40)
    print("  Bharat Benefits - Voice Agent")
    print(f"  mode: {mode_label}")
    print("  say 'exit' or press Ctrl+C to stop.")
    print("-" * 40)

    while True:
        turn += 1
        print(f"\n[turn {turn}]")

        try:
            # -- Input --------------------------------------------------------
            if text_mode:
                try:
                    question = input("  question: ").strip()
                    lang = None
                except (KeyboardInterrupt, EOFError):
                    break
            elif use_vad:
                audio_path = record_vad()
                if audio_path is None:
                    print("  nothing detected, try again.")
                    continue
                question, lang = transcribe(audio_path)
            else:
                audio_path = record_fixed(record_seconds)
                question, lang = transcribe(audio_path)

            if not question:
                print("  nothing heard, try again.")
                continue

            # -- Exit check ---------------------------------------------------
            if question.lower().strip().rstrip("?.!") in EXIT_PHRASES:
                print("\nGoodbye.")
                speak("Goodbye. Thank you for using Bharat Benefits.")
                break

            # -- RAG + LLM + TTS ----------------------------------------------
            answer = agent.process_turn(question)
            if text_mode:
                print(f"  answer: {answer}")
            else:
                speak(answer, target_language=lang)

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            print("Retrying next turn ...")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Bharat Benefits Voice Agent")
    parser.add_argument(
        "--seconds",
        type=int,
        default=5,
        help="Recording window in seconds for fixed mode (default: 5)",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Use voice activity detection - stops recording when you pause",
    )
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="Type questions instead of speaking (for debugging)",
    )
    args = parser.parse_args()

    import asyncio
    asyncio.run(
        live_loop(
            record_seconds=args.seconds,
            use_vad=args.vad,
            text_mode=args.text_mode,
        )
    )


if __name__ == "__main__":
    main()
