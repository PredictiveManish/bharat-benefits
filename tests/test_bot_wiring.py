"""The agent answers from the local index when Moss is not configured.

bot.py imports the audio stack at module level, so the audio libraries are
stubbed here; everything the retrieval path touches is real.
"""

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub the audio dependencies: this test exercises retrieval, not playback.
for name in ("sounddevice", "soundfile", "simpleaudio", "webrtcvad"):
    sys.modules.setdefault(name, types.ModuleType(name))

import pytest

bot = pytest.importorskip("bot")


@pytest.fixture(scope="module")
def index():
    from bharat.retrieval import LocalIndex

    return LocalIndex.from_directory(bot.SCHEMES_DIR)


def test_retrieve_uses_the_local_index(index):
    context, sources = asyncio.run(
        bot.retrieve("How much money does PM-KISAN give?", None, index)
    )
    assert sources == ["PM-KISAN"]
    assert "[Source: PM-KISAN" in context
    assert "6,000" in context


def test_retrieve_reports_nothing_for_an_off_topic_question(index):
    context, sources = asyncio.run(
        bot.retrieve("What is the weather in Mumbai tomorrow?", None, index)
    )
    assert sources == []
    assert "No relevant information" in context


def test_retrieve_without_any_retriever_fails_loudly():
    with pytest.raises(RuntimeError):
        asyncio.run(bot.retrieve("anything", None, None))
