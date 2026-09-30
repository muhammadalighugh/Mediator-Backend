# Argument Mediator — Backend

> **GitHub:** [github.com/muhammadalighugh/Mediator-Backend](https://github.com/muhammadalighugh/Mediator-Backend)

FastAPI server that powers live transcription, claim extraction, evidence matching, contradiction detection, and final mediation report generation.

---

## Tech stack

| Layer | Technology |
|---|---|
| API server | FastAPI + Uvicorn |
| Realtime transcription | AssemblyAI Streaming v3 |
| Batch diarization | AssemblyAI batch API |
| LLM reasoning | OpenAI **or** Anthropic (switchable via env) |
| Vector store | ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`) |
| PDF parsing | pypdf |

---

## Project structure

```
backend/
├── main.py                  # FastAPI app, CORS, router wiring
├── requirements.txt
├── .env.example             # Copy to .env and fill in keys
│
├── api/
│   ├── http_routes.py       # POST /upload-evidence, GET /report, GET /session
│   └── ws_routes.py         # WebSocket /ws/{session_id}
│
├── core/
│   ├── config.py            # Pydantic settings (reads .env)
│   └── session.py           # Session + SessionManager (in-memory)
│
├── evidence/
│   ├── ingest.py            # Parse + chunk txt/md/csv/pdf into EvidenceChunks
│   ├── evidence_matcher.py  # Batched LLM verdict matching + fabrication guard
│   └── vector_store.py      # ChromaDB wrapper
│
├── models/
│   ├── enums.py             # StatementType, VerdictType, SessionStatus
│   └── schemas.py           # Pydantic models (Claim, EvidenceChunk, MediationReport …)
│
├── reasoning/
│   ├── claim_extractor.py   # LLM claim extraction + live coordinator
│   ├── contradiction_detector.py  # Conflict + agreement + dispute type detection
│   ├── mediation_report.py  # Full end-session pipeline (diarize → extract → match → summarise)
│   ├── clarifying_questions.py
│   └── llm_client.py        # Unified OpenAI / Anthropic client
│
├── transcription/
│   ├── assemblyai_client.py # Streaming v3 session wrapper
│   ├── speaker_mapper.py    # AssemblyAI label → speaker_id resolution
│   └── stream_handler.py    # TurnEvent → Utterance → WebSocket broadcast
│
├── sessions/                # Runtime data (ignored by Git)
│   └── .gitkeep
└── tests/
    ├── reasoning_smoke_test.py
    └── manual_test_client.py
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- An [AssemblyAI](https://www.assemblyai.com/) API key
- An [OpenAI](https://platform.openai.com/) or [Anthropic](https://console.anthropic.com/) API key

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on first run.

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
LLM_PROVIDER=openai          # or "anthropic"
LLM_API_KEY=your_llm_api_key_here
LLM_MODEL=                   # leave blank for provider default
SAMPLE_RATE=16000
```

### 5. Run the server

```bash
uvicorn main:app --reload --port 8000
```

The API is now available at `http://localhost:8000`.

---

## API reference

### WebSocket — `ws://localhost:8000/ws/{session_id}`

| Direction | Message type | Payload |
|---|---|---|
| Client → Server | `start_session` | `{ speakers: [string, string] }` |
| Client → Server | `audio_chunk` | `{ data: base64 PCM bytes }` |
| Client → Server | `end_session` | `{}` |
| Server → Client | `transcript_partial` | `{ speaker_id, text, start_ms, end_ms }` |
| Server → Client | `transcript_final` | `{ speaker_id, text, start_ms, end_ms, utterance_id }` |
| Server → Client | `claims_updated` | `{ claims: Claim[] }` |
| Server → Client | `report_ready` | `{ report: MediationReport }` |

### REST

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload-evidence/{session_id}` | Upload a `.txt / .md / .csv / .pdf` evidence file |
| `GET` | `/report/{session_id}` | Fetch the final `MediationReport` JSON |
| `GET` | `/session/{session_id}` | Debug dump of session state |
| `GET` | `/` | Health check → `{ "status": "ok" }` |

---

## Pipeline (end_session flow)

```
end_session received
        │
        ▼
(a) Diarize WAV  ──────────────────────────────►  canonical Utterance list
        │
        ▼
(b) Clear live claims → re-extract on canonical transcript
        │
        ▼
(c) Detect contradictions / agreements / dispute type
        │
        ▼
(d) Match claims against evidence corpus
        │   └─ validate_verdicts(): downgrade fabricated quotes → UNCERTAIN
        ▼
(e) Generate executive summary (LLM)
        │
        ▼
(f) Assemble & store MediationReport → broadcast report_ready
```

---

## Claim types

| Type | Meaning |
|---|---|
| `claim` | Assertion that needs evidence support |
| `fact` | Directly verifiable statement (also evidence-matched) |
| `assumption` | Unstated premise treated as self-evident |
| `evidence_ref` | Reference to a document / log / message thread |
| `opinion` | Value judgment — skipped during evidence matching |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ASSEMBLYAI_API_KEY` | ✅ | AssemblyAI key for transcription + diarization |
| `LLM_PROVIDER` | ✅ | `openai` or `anthropic` |
| `LLM_API_KEY` | ✅ | Key for the chosen LLM provider |
| `LLM_MODEL` | ➖ | Override model name (blank = provider default) |
| `SAMPLE_RATE` | ➖ | Audio sample rate in Hz (default `16000`) |
| `MONGODB_URI` | ✅ | MongoDB Atlas connection string (required for auth) |
| `JWT_SECRET` | ✅ | Long random string used to sign session tokens |
| `ALLOWED_ORIGINS` | ➖ | Comma-separated CORS origins (default `*`; set to Vercel URL in prod) |

---

## Frontend

The Next.js frontend is deployed separately on **Vercel**:

- **GitHub:** [github.com/muhammadalighugh/Mediator-Front](https://github.com/muhammadalighugh/Mediator-Front)
- **Live URL:** `https://mediator-front-psi.vercel.app`
