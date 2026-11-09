# Bharat Benefits Voice Agent

A voice-first RAG assistant for Indian public-benefit schemes, built with
ChromaDB semantic retrieval and Sarvam AI speech APIs.

---

## What This Does

Users ask questions by voice or text about Indian government welfare schemes
and receive spoken, grounded answers in return.

```
voice / text input
    |
Sarvam STT  (saaras:v3)         converts speech to text
    |
ChromaDB retrieval              finds relevant scheme snippets from local knowledge base
    |
Sarvam Chat Completion          generates a grounded, voice-friendly answer
    |
Sarvam TTS  (bulbul:v3)         converts the answer to speech
    |
audio response                  played back through speakers
```

The knowledge base covers five schemes:

- Ayushman Bharat / PM-JAY - health insurance for poor families
- PM-KISAN - income support for small farmers
- PM Vishwakarma - skill and financial support for artisans
- National Scholarship Portal - centralised scholarship access for students
- PMKVY / Skill India - free vocational skill training and certification

> This is a demo knowledge base, not an official government service. Always
> verify eligibility, amounts, and deadlines on the official government portals
> listed in each scheme file.

---

## Recording Modes

The agent supports three input modes:

**Fixed window (default)**
Records for a set number of seconds per turn. Simple and predictable.

```bash
python main.py              # 5 seconds per turn
python main.py --seconds 8  # 8 seconds per turn
```

**VAD mode** (recommended for natural conversation)
Uses voice activity detection to automatically stop recording when you
pause speaking. No need to worry about a timer — just speak naturally and
the agent picks up when you stop.

```bash
python main.py --vad
```

**Text mode** (for debugging)
Type questions instead of speaking. Useful when testing without a microphone.

```bash
python main.py --text-mode
```

---

## Why ChromaDB?

ChromaDB is an open-source embedding database designed for AI applications.

- Fully self-hosted, no cloud dependency or API keys needed.
- Persistent storage on local disk.
- Built-in cosine similarity search.
- First-class support for metadata filtering.
- Automatic embedding generation with sentence-transformers.
- Zero per-query costs — runs entirely on your machine.

ChromaDB retrieves the top-3 most relevant scheme snippets for each question,
which are passed as grounded context to the LLM. This prevents hallucination
of eligibility rules, amounts, or deadlines.

### ChromaDB features used

| Feature | Purpose |
|---|---|
| `PersistentClient(path)` | Store embeddings on disk |
| `collection.upsert()` | Index documents with embeddings |
| `collection.query()` | Semantic search with metadata |
| `SentenceTransformer` | Generate embeddings locally |

---

## Why Sarvam?

Sarvam AI is India's purpose-built AI stack for Indian languages.

- Saaras v3 (STT): speech-to-text with support for 23 Indian languages including
  Hindi, Bengali, Tamil, Telugu, Punjabi, and more.
- Sarvam-30B / 105B (Chat): multilingual chat models with Indian linguistic context.
- Bulbul v2/v3 (TTS): natural-sounding Indian-language voices.

The combination handles queries in multiple Indian languages out of the box.

---

## Setup

### 1. Navigate to the project folder

```bash
cd bharat-benefits/
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

On Debian/Ubuntu, if `simpleaudio` fails to build:
```bash
sudo apt-get install libasound2-dev
```

`webrtcvad` is only required for `--vad` mode. If you don't need it, the
rest of the agent works without it.

The first run will automatically download the embedding model
(`all-MiniLM-L6-v2`, ~80 MB).

### 4. Set environment variables

```bash
cp .env.example .env
```

Fill in:

| Variable | Where to get it |
|---|---|
| `SARVAM_API_KEY` | dashboard.sarvam.ai |

ChromaDB requires no credentials — it stores data locally.

### 5. Create the knowledge base (run once, or after adding new schemes)

```bash
python ingest.py
```

Expected output:

```
Loading scheme documents from: data/schemes/
   Found 5 document(s):
   - ayushman-bharat.md (Ayushman Bharat)
   - national-scholarship-portal.md (National Scholarship Portal)
   - pm-kisan.md (Pm Kisan)
   - pm-vishwakarma.md (Pm Vishwakarma)
   - pmkvy-skill-india.md (Pmkvy Skill India)
   Generating embeddings ...
   Ingested 23 chunk(s) from 5 document(s).

Knowledge base ready:
   Collection : bharat-benefits
   Documents  : 23 chunk(s)
   Path       : ./chroma_data
```

---

## Running the Agent

```bash
# Fixed window, 5 seconds per turn (default)
python main.py

# Fixed window, custom duration
python main.py --seconds 8

# VAD mode - stops when you pause (most natural)
python main.py --vad

# Text mode - type questions (no mic needed)
python main.py --text-mode
```

---

## Example Questions

| Question | Relevant scheme |
|---|---|
| "I am a small farmer. Which scheme can help me?" | PM-KISAN |
| "What documents are needed for Ayushman Bharat?" | Ayushman Bharat |
| "I am a carpenter. Is there any government scheme for me?" | PM Vishwakarma |
| "How much money does PM-KISAN give, and how often?" | PM-KISAN |
| "I want free vocational training. What should I do?" | PMKVY |
| "My family is poor and needs medical coverage." | Ayushman Bharat |
| "I am a student looking for scholarship support." | NSP |
| "Which scheme is useful for skill training?" | PMKVY / Skill India |

---

## Environment Variables

```
SARVAM_API_KEY          Sarvam AI subscription key (required)
SARVAM_STT_MODEL        STT model, default: saaras:v3 (optional)
SARVAM_CHAT_MODEL       Chat model, default: sarvam-105b (optional)
SARVAM_TTS_MODEL        TTS model, default: bulbul:v3 (optional)
SARVAM_TTS_SPEAKER      TTS speaker voice, default: priya (optional)
CHROMA_DB_PATH          ChromaDB storage path, default: ./chroma_data (optional)
CHROMA_COLLECTION       Collection name, default: bharat-benefits (optional)
EMBEDDING_MODEL         Embedding model, default: all-MiniLM-L6-v2 (optional)
TOP_K                   Results per query, default: 3 (optional)
CHUNK_SIZE              Max chars per chunk, default: 500 (optional)
CHUNK_OVERLAP           Overlap between chunks, default: 50 (optional)
```

---

## Folder Structure

```
bharat-benefits/
├── README.md
├── .env.example
├── requirements.txt
├── ingest.py            knowledge base ingestion script
├── main.py              main voice agent (STT -> ChromaDB -> LLM -> TTS)
├── bharat_benefits/
│   ├── __init__.py
│   └── services/
│       ├── __init__.py
│       ├── knowledge_base.py  ChromaDB retrieval
│       ├── agent.py           LLM interaction
│       └── audio.py           recording & playback
├── data/
│   └── schemes/
│       └── central/
│           ├── ayushman-bharat.md
│           ├── pm-kisan.md
│           ├── pm-vishwakarma.md
│           ├── national-scholarship-portal.md
│           └── pmkvy-skill-india.md
└── chroma_data/         created by ingest.py (vector store)
```

---

## Limitations

- Demo knowledge base only. Scheme information may not reflect the latest
  government guidelines, benefit amounts, or eligibility criteria.
- Not an official government service. Cannot process applications, verify
  beneficiary status, or access real-time government databases.
- Audio playback requires `simpleaudio`. If unavailable, audio is saved to disk.
- VAD mode requires `webrtcvad`. Falls back to fixed-window if not installed.
- Embedding model is downloaded on first run (~80 MB).

Official portals for verification:

- Ayushman Bharat: https://pmjay.gov.in
- PM-KISAN: https://pmkisan.gov.in
- PM Vishwakarma: https://pmvishwakarma.gov.in
- National Scholarship Portal: https://scholarships.gov.in
- Skill India / PMKVY: https://www.skillindia.gov.in

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `chromadb` | >=0.5.0 | Open-source vector database |
| `sentence-transformers` | >=3.0.0 | Embedding generation |
| `python-dotenv` | >=1.0.0 | Load .env credentials |
| `httpx` | >=0.27.0 | HTTP client for Sarvam API calls |
| `sounddevice` | >=0.4.6 | Microphone recording and playback |
| `soundfile` | >=0.12.1 | WAV file read/write |
| `simpleaudio` | >=1.0.4 | Audio playback (optional) |
| `numpy` | >=1.26.0 | Audio array handling |
| `webrtcvad` | >=2.0.10 | Voice activity detection for --vad mode (optional) |

---

Built for the Moss Live Labs community.
