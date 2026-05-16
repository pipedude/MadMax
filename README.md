# 😎 MadMax Live Agent

A voice agent for devices, powered by **Google Gemini Live API** with a local wake-word detector and long-term memory.

**Capabilities:**
- 🗣️ Realtime Speech-to-Speech dialogue via Gemini Live API
- 🔍 Fresh information from the internet via Google Search (grounding)
- 💰 Budget-friendly: offline mode with local wake-word (Vosk) and auto-shutdown timer
- 🧠 Managed long-term memory based on JSON files: people, places, facts, goals, experience, episodes, reflections and persona
- 🛠️ Tool calling in live mode

**Technical highlights:**
- 🔒 Transactional memory isolation: backup + rollback on errors, single-use guard
- 🧪 LLM Surgeon: automatic memory conflict resolution via LLM
- 📝 Auto-save all sessions to daily markdown
- 🔄 Automatic recovery of missed sessions
- ⏱️ Graceful shutdown with configurable timeouts
- 📊 Latency diagnostics for all LLM calls in logs

**In development:**
- 📷 Photo and video stream processing
- 🧹 Smart long-term memory cleanup
- 🤖 Integration with ROS2 modules for robot control
- 🔧 Other integrations and improvements

---

## 🚀 Key Features

### 💬 Realtime Voice Loop
- Local wake-word detection via **Vosk** (no LLM costs)
- Instant transition to live mode on the wake word
- Speech-to-Speech dialogue with minimal latency

### 🧠 Post-Session Memory Pipeline
- Automatic extraction: facts, goals, experience, episodes, reflections and persona after every session
- LLM Surgeon: memory conflict resolution (UPDATE / MERGE / APPEND / IGNORE) via a separate LLM call
- Rebuild of `active_context.json` for upcoming dialogues
- Automatic recovery of unprocessed sessions

### 🎭 Agent Persona (Max)
- Name, gender and communication style are set in `agent_instructions.md`
- `SOUL.md` — philosophical persona manifesto: attitude toward the world, inclinations, shadow, meta-reflection, written by the agent itself after hours of testing conversations
- Automatic extraction of reflections and persona traits from dialogues into `reflections.json`

---

## 🏗️ Architecture

### 1️⃣ Sleep Mode
Agent is offline; microphone is monitored locally via **Vosk**. No LLM costs.

### 2️⃣ Active Session
After the wake word audio switches to **Gemini Live API**. Dialogue runs in realtime.

### 3️⃣ Post-Session Processing
After a session ends:
1. Save transcript to `memory_engine/daily/YYYY-MM-DD.md`
2. Call `process_missing_sessions(day_date)` — process all unprocessed sessions of the day
3. Call `build_memory_context()` — rebuild active context

---

## 💾 Memory Structure

### 📅 Daily Markdown Logs
Every session is saved to `memory_engine/daily/YYYY-MM-DD.md` with metadata:
- `session_id`
- `started_at` and `ended_at` (ISO 8601 with timezone offset)
- Dialogue transcript

**Source of truth** for post-session processing.

### 🧩 Active Context
File `memory_engine/active_context.json`:
```json
{
  "last_context": "...",
  "summary_yesterday": "...",
  "summary_today": "...",
  "reply_count_today": 0,
  "summary_reply_count": 0,
  "long_term_injections": []
}
```

Used as the working context for upcoming live sessions.

### 🗄️ Long-Memory Storage
Directory `memory_engine/memory/`:
- `people.json` — information about people
- `places.json` — places and locations
- `facts.json` — facts and knowledge
- `goals.json` — goals and tasks
- `experience.json` — experience and skills
- `reflections.json` — reflections and insights
- `episodes.log.jsonl` — episode chronology
- `processed_sessions.json` — registry of processed sessions (prevents reprocessing the same session)

---

## ⚙️ Configuration

Google AI Studio API Key in `.env`
Key parameters in `config.py`
Memory settings in `memory_config.py`

---

## 📂 Project Structure

```
MadMax/
├── main.py                          # Entry point
├── config.py                        # Configuration
├── agent_instructions.md            # Agent system prompt (Max)
├── SOUL.md                          # Agent persona and philosophy
├── core/
│   ├── orchestrator.py              # Agent lifecycle (sleep / live / post-session)
│   ├── audio_io.py                  # Audio I/O and Vosk wake-word
│   ├── gemini_client.py             # Gemini Live API client
│   ├── agent_tools.py               # Memory tools for live mode
│   ├── session_transcript_logger.py # Session transcript persistence
│   ├── errors.py                    # Exceptions
│   └── state.py                     # Session state
├── memory_engine/
│   ├── active_context_builder.py    # active_context.json builder
│   ├── long_memory_extractor_agent.py  # Memory extraction from transcripts
│   ├── long_memory_apply.py         # Memory operations + LLM Surgeon
│   ├── long_memory_normalize.py     # Normalization and fuzzy matching
│   ├── long_memory_ops.py           # Operation schemas and validation
│   ├── long_memory_query_service.py # Long-term memory search
│   ├── summarize_context_agent.py   # Day summarization
│   ├── llm_client_utils.py          # LLM timeout and diagnostics
│   ├── memory_config.py             # Memory paths and constants
│   ├── entity_policies.py           # Entity link policies
│   ├── time_policy.py               # Timestamp policy
│   ├── daily/                       # Daily markdown logs
│   └── memory/                      # Long-memory JSON files + backups
└── live_api_docs/                   # Gemini Live API documentation
```

---

## 🚀 Quick Start

```bash
# 1. Clone and enter directory
cd MadMax

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download wake-word model (~40 MB)
./setup.sh

# 5. Configure environment
# Create .env file (or export variables):
# GOOGLE_API_KEY=your_key_here

# 6. Run the agent
python main.py
```

**Requirements:**
- Python 3.11+
- Microphone and speakers
- Google AI Studio API Key

---

## 🛠️ Roadmap

### ✅ Already implemented
- **Function Calling for memory** — live agent calls `memory_lookup_person`, `memory_lookup_goal`, `memory_lookup_experience`, `memory_recent_episodes` during dialogue
- **Google Search (grounding)** — agent receives fresh information from the internet in realtime
- **Transactional memory isolation** — backup before write, rollback on errors, single-use guard for `apply_payload`
- **LLM Surgeon** — automatic memory conflict resolution via a separate LLM call with batching
- **Fail-fast error handling** — explicit logs on corrupted JSON, graceful `CancelledError`, latency diagnostics

### 🎯 Planned

### 🧹 Smart long-term memory cleanup
**Goal:** Automatic removal of stale or irrelevant data from memory.

**Planned logic:**
- **Fact prioritization** — relevance score based on access frequency and freshness
- **Old episode archival** — move rarely used episodes to cold storage
- **Automatic duplicate merging** — find and merge similar facts/goals
- **Temporary goal expiration** — auto-complete or archive goals with expired deadlines
- **Configurable retention rules** — set data lifetime for different categories

**Result:** Memory stays relevant, does not grow uncontrollably, and is not cluttered with duplicates and outdated information.

### 🔧 Refactoring & Type Safety
- **Pydantic** for structured payloads instead of `dict[str, Any]`

---

## 🏗️ Technical Debt

Consciously accepted trade-offs that are known and documented:

| Issue | Impact | Why we kept it |
|-------|--------|----------------|
| **`Any` instead of Pydantic** for operation payloads | No type safety, IDE does not suggest fields | It works, changing it requires rewriting 5+ modules |
| **Tight coupling: `GeminiLiveClient` imports `AudioIO`** | Hard to test, risk of circular dependency | No DI container, Protocols require refactoring |
| **No CI/CD** | No automatic type checking and tests | Project is developed locally, pytest is run manually |

---

## 🤖 Agentic Engineering

**Important:** A significant part of this project was written using **Agentic Engineering** in pair-programming mode.

---

## 📊 Current Status

The project consists of three stable loops:
- **Live conversation loop** — realtime Speech-to-Speech dialogue with the user (Google Search, tool calling, Vosk wake-word)
- **Post-session memory loop** — automatic extraction, deduplication and knowledge persistence (people, places, facts, goals, experience, episodes, reflections, persona)
- **Reliability loop** — transactional isolation (backup + rollback), graceful shutdown, LLM timeouts, recovery of missed sessions

The voice agent is ready for daily use as-is. The main constraints are architectural debt (see section above), not functional issues.