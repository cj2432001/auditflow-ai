# AuditFlow AI Agent — Smart Travel & Expense Concierge

> **Track: Agents for Business** · Google 5-Day AI Agents Intensive 2026 Capstone Project

AuditFlow AI is an enterprise-grade AI agent that automates corporate travel expense routing, policy compliance auditing, and fraud detection — built with Google's Agent Development Kit (ADK) and Gemini 3.5 Flash.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **ADK Multi-Node Workflow** | Stateful 7-node pipeline: parse → security → route → LLM audit → human review → record |
| 🛡 **Prompt Injection Guard** | Intercepts adversarial descriptions before they reach the LLM |
| 🔒 **PII Scrubbing** | Auto-redacts SSNs and credit card numbers from expense descriptions |
| ⚡ **LLM Risk Audit** | Gemini 3.5 Flash scores expenses 1–10, flags violations, and recommends approve/reject |
| 👥 **Human-in-the-Loop** | ADK `RequestInput` interrupt suspends the session for manager review; resumes on decision |
| 📊 **Web Dashboard** | Premium dark UI — live submission form, audit trail, real-time stats |
| 🧪 **Full Test Suite** | 13 automated tests: unit, integration, and E2E server tests |

---

## 🏗 Architecture

```
Submit Expense
      │
      ▼
 parse_expense         ← Handles JSON / base64 / plain text
      │
      ▼
 security_checkpoint   ← PII scrub + prompt injection detection
      │
   ┌──┴──────────────────────┐
   │ injection               │ clean
   ▼                         ▼
human_approval          route_expense
   │                         │
   │              ┌──────────┴──────────┐
   │              │ amount < $100       │ amount ≥ $100
   │              ▼                     ▼
   │         record_outcome         llm_review (Gemini)
   │                                    │
   │                               post_llm_routing
   │                             ┌──────┴──────────────┐
   │                             │ low risk             │ high risk
   │                             ▼                      ▼
   │                        record_outcome         human_approval
   │                                                    │
   └────────────────────────────────────────────────────┘
                                                         │
                                                    record_outcome
```

---

## 🛠 Tech Stack

- **Agent Framework:** Google Agent Development Kit (ADK)
- **LLM:** `gemini-3.5-flash`
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Vanilla HTML5 / CSS3 / JavaScript (no framework dependencies)
- **Testing:** Pytest (unit, integration, E2E)
- **Package Manager:** `uv`

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- A [Google AI Studio](https://aistudio.google.com/) API key

### Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/auditflow-ai.git
cd auditflow-ai

# 2. Install dependencies
uv sync

# 3. Set your Gemini API key
export GEMINI_API_KEY=your-api-key-here

# 4. Start the server
uv run python -m app.fast_api_app
```

5. Open your browser at **[http://localhost:8000/dashboard/](http://localhost:8000/dashboard/)**

---

## 🧪 Running Tests

```bash
uv run pytest tests/ -v
```

Expected output: **13 passed** across unit, integration, and E2E suites.

---

## 📁 Project Structure

```
capstone-project/
├── app/
│   ├── agent.py              # ADK workflow: all nodes, schemas, routing logic
│   ├── fast_api_app.py       # FastAPI backend + /api/submit & /api/respond endpoints
│   ├── static/
│   │   └── index.html        # Premium web dashboard UI
│   └── app_utils/            # Session, artifact, telemetry, A2A helpers
├── tests/
│   ├── unit/                 # Node-level unit tests (no network)
│   ├── integration/          # ADK runner integration tests
│   └── integration/test_server_e2e.py  # Full server E2E tests
├── .env.example              # Template — copy to .env and add your API key
└── pyproject.toml            # Project dependencies
```

---

## 🔒 Security & Privacy

- **No API keys are committed** — `.env` is gitignored; use `.env.example` as a template
- **PII protection** runs on every submission before any LLM call
- **Prompt injection** is caught at the security checkpoint node and never reaches Gemini


