"""
Audio service - handles microphone recording and TTS playback.
"""

import asyncio
import base64
import collections
import os
import tempfile
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf

# -- Config -------------------------------------------------------------------

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE = "https://api.sarvam.ai"
STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "priya")
SAMPLE_RATE = 16_000  # Hz - required by Sarvam STT

# VAD settings
VAD_FRAME_MS = 30  # ms per frame
VAD_AGGRESSIVENESS = 2  # 0 (least) to 3 (most aggressive)
VAD_SILENCE_TIMEOUT = 1.2  # seconds of silence before stopping
VAD_MAX_SECONDS = 15  # hard cap


def record_fixed(seconds: int = 5) -> Path:
    """Record a fixed number of seconds from the microphone."""
    print(f"\nListening ({seconds}s) ...")
    try:
        audio = sd.rec(
            int(seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()
    except Exception as e:
        print(f"ERROR: microphone unavailable: {e}", file=__import__('sys').stderr)
        raise

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE, subtype="PCM_16")
    tmp.close()
    return Path(tmp.name)


def record_vad() -> Path:
    """
    Record until the speaker stops talking using VAD.
    Stops after VAD_SILENCE_TIMEOUT seconds of silence or VAD_MAX_SECONDS cap.
    """
    try:
        import webrtcvad
    except ImportError:
        print(
            "ERROR: webrtcvad not installed. Run: pip install webrtcvad",
            file=__import__('sys').stderr,
        )
        raise

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    frame_samples = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
    frame_bytes = frame_samples * 2
    silence_frames = int(VAD_SILENCE_TIMEOUT * 1000 / VAD_FRAME_MS)
    max_frames = int(VAD_MAX_SECONDS * 1000 / VAD_FRAME_MS)

    ring = collections.deque(maxlen=silence_frames)
    frames_recorded = []
    triggered = False
    silent_count = 0

    print("\nListening (speak now, will stop when you pause) ...")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=frame_samples
    ) as stream:
        total = 0
        while total < max_frames:
            raw, _ = stream.read(frame_samples)
            frame = bytes(raw)
            if len(frame) < frame_bytes:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            total += 1

            if not triggered:
                ring.append((frame, is_speech))
                num_voiced = sum(1 for _, s in ring if s)
                if num_voiced > 0.5 * ring.maxlen:
                    triggered = True
                    frames_recorded.extend(f for f, _ in ring)
                    ring.clear()
                    silent_count = 0
            else:
                frames_recorded.append(frame)
                if is_speech:
                    silent_count = 0
                else:
                    silent_count += 1
                    if silent_count > silence_frames:
                        break

    if not frames_recorded:
        return None

    audio_bytes = b"".join(frames_recorded)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_array, SAMPLE_RATE, subtype="PCM_16")
    tmp.close()
    return Path(tmp.name)


def transcribe(audio_path: Path) -> tuple[str, str]:
    """
    Send audio to Sarvam STT. Returns (transcript, detected_language).
    """
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    resp = httpx.post(
        f"{SARVAM_BASE}/speech-to-text",
        headers={"api-subscription-key": SARVAM_API_KEY},
        files={"file": (audio_path.name, audio_bytes, "audio/wav")},
        data={"model": STT_MODEL, "language_code": "unknown"},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    transcript = data.get("transcript", "").strip()
    lang = data.get("language_code", "unknown")

    try:
        audio_path.unlink()
    except Exception:
        pass

    return transcript, lang


def speak(answer: str, target_language: str = "en-IN") -> None:
    """Convert text to speech and play it back."""
    text = answer[:MAX_TTS_CHARS]
    print("   speaking ...")

    resp = httpx.post(
        f"{SARVAM_BASE}/text-to-speech",
        headers={
            **{"api-subscription-key": SARVAM_API_KEY},
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "target_language_code": target_language,
            "speaker": TTS_SPEAKER,
            "model": TTS_MODEL,
            "enable_preprocessing": True,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    audios = resp.json().get("audios", [])
    if not audios:
        print("   WARNING: TTS returned no audio.")
        return

    wav_bytes = base64.b64decode(audios[0])

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_bytes)
    tmp.close()

    try:
        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(tmp.name)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except ImportError:
        data, sr = sf.read(tmp.name, dtype="int16")
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"   WARNING: playback failed: {e}")
        return

    try:
        Path(tmp.name).unlink()
    except Exception:
        pass
