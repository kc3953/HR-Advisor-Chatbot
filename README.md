# HR Advisor — People Analytics Chatbot

[![CI](https://github.com/kc3953/HR-Advisor-Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/kc3953/HR-Advisor-Chatbot/actions/workflows/ci.yml)

A domain-specific chatbot that answers HR and people analytics questions, enforces strict scope boundaries, and handles distressed users safely. Also includes a live text-to-SQL agent and a people analytics dashboard, both backed by a real HR dataset. Built with FastAPI + Gemini on Vertex AI, deployed on Google Cloud Run.

[**Live URL Link**](https://hr-advisor-36222488155.us-central1.run.app/static/index.html) &bull; [Dashboard](https://hr-advisor-36222488155.us-central1.run.app/static/dashboard.html) &bull; [Ask AI](https://hr-advisor-36222488155.us-central1.run.app/static/ask.html) &bull; [Insights Memo](https://hr-advisor-36222488155.us-central1.run.app/static/insights.html)

## What It Does

- Answers conceptual questions about HR policies, employee benefits, performance management, talent acquisition, workforce planning, and people analytics concepts
- **Answers quantitative questions with real computed numbers**: a text-to-SQL agent generates and safely executes SQL against a real HR dataset (e.g. "What's the attrition rate by department?"), then narrates the result as an answer + insight + recommendation
- **People Analytics Dashboard** (`/static/dashboard.html`): fixed KPI charts — headcount trend, attrition rate by department, hires by recruitment source, and attrition rate by manager (a self-join, see **Bugs I Found and Fixed** below) — with filters, computed live from the dataset
- **Ask AI** (`/static/ask.html`): a standalone page where any free-text question generates its own chart (bar/line/pie, picked by the LLM) and narration; each question stays in a running history instead of replacing the last result
- **Insights Memo** (`/static/insights.html`): three synthesized findings (Finding → So What → Recommendation), the "what I'd bring to a leadership review" artifact distinct from the interactive tools
- Refuses out-of-scope topics (technology/coding, lifestyle, financial/legal advice) with a clear explanation
- Detects distressed users and immediately provides crisis resources (988, Crisis Text Line, EAP)
- Uses a Python backstop classifier to catch cases where the LLM misses scope or safety rules

## Data Source

The SQL agent and dashboard run on **[HRDataset_v14](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set)** by Rich Huebner (CC0 Public Domain). It's a widely-used reference dataset of a fictitious company's HR records (311 employees) with real hire/termination dates, department, performance, and recruitment-source fields — no real individuals' data is used.

## Bugs I Found and Fixed

A few real bugs surfaced while building and testing this against live data — each is covered by a regression test so it can't silently come back.

1. **Data-question routing swallowed conceptual questions.** Keyword-based routing sent "What is employee attrition rate and how is it calculated?" to the SQL agent (it matched "attrition rate"), when it should get a conceptual explanation instead. Caught by running the existing eval harness after adding the SQL agent. Fixed by adding a `CONCEPTUAL_MARKERS` guard (`app/main.py`) that checks for phrasing like "how is it calculated" before routing to SQL.
2. **Headcount trend approximated every month-end as the 28th.** The original implementation used `f"{month}-28"` for every month regardless of actual length, silently misattributing late-month headcount changes (29th-31st) to the following month. Fixed with `calendar.monthrange()` (`app/main.py`).
3. **A self-join returned zero rows on the first attempt.** Building the manager-attrition chart, `employees.manager_name = employees.employee_name` matched nothing — the two columns turn out to use different name orderings (`"Last, First"` vs. `"First Last"`). Fixed by adding a normalized `employee_display_name` column at load time (`app/db.py`) and joining on that instead. Both the zero-match failure and the fixed join are covered by regression tests (`tests/test_db.py`) so this can't quietly break again.
4. **Ask-AI chart validation didn't check that `y_field` was numeric.** A real live question ("Which recruitment source has the most hires?") produced SQL that only selected the category column, and the LLM's chart spec set `x_field == y_field`, rendering category names as chart "values" instead of numbers. Fixed by validating `y_field` is numeric and distinct from `x_field` before accepting a chart spec (`app/sql_agent.py`), with regression tests covering both the bad and good cases.

## Rate Limiting

This service is public and unauthenticated, and every `/chat` or `/api/dashboard/ask` call spends real Vertex AI budget, so requests are throttled per IP address (via [slowapi](https://github.com/laurentS/slowapi)):

| Endpoint(s) | Limit |
|---|---|
| `/chat`, `/api/dashboard/ask` (LLM calls) | 10 requests/minute |
| `/api/dashboard/headcount`, `/attrition`, `/recruitment-source`, `/filter-options` (pure SQL) | 60 requests/minute |

Cloud Run is also capped at `--max-instances=2`, which bounds worst-case concurrent cost regardless of rate-limiting behavior.

For local development, `eval.py` fires ~40 sequential requests at `/chat`, which would otherwise trip the 10/minute limit. Disable rate limiting for local eval runs with:
```bash
RATE_LIMIT_ENABLED=false uv run python eval.py
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.5 Flash via Vertex AI |
| Backend | FastAPI (Python 3.12) |
| Data | SQLite (in-memory), loaded from a real HR dataset |
| Frontend | Vanilla HTML/CSS/JS, Chart.js (dashboard) |
| Deployment | Google Cloud Run |
| Package manager | uv |

---

## Project Structure

```
domain-chatbot/
├── app/
│   ├── main.py           # FastAPI app, system prompt, backstop classifier, dashboard API
│   ├── db.py             # Loads HRDataset_v14.csv into an in-memory SQLite DB
│   ├── sql_agent.py       # Text-to-SQL generation, validation, execution, narration
│   ├── data/
│   │   └── HRDataset_v14.csv
│   └── static/
│       ├── index.html     # Chat UI
│       ├── dashboard.html # Fixed KPI charts + filters (Chart.js)
│       ├── ask.html       # Standalone ask-a-question -> chart history feed
│       └── insights.html  # Synthesized findings memo (Finding / So What / Recommendation)
├── tests/                # Unit tests (pytest) — SQL safety, dataset loading, filters
├── eval.py              # Evaluation harness (deterministic + MaaJ)
├── eval_dataset.json    # 40 test cases across 4 categories
├── Dockerfile
├── cloudbuild.yaml
├── pyproject.toml
└── .gcloudignore
```

---

## Run Locally

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) installed
- Google Cloud SDK installed and authenticated
- A GCP project with Vertex AI enabled

### 1. Add gcloud to your PATH
```bash
export PATH="/path/to/google-cloud-sdk/bin:$PATH"
```

### 2. Authenticate with GCP
```bash
gcloud auth application-default login
gcloud config set project YOUR-PROJECT-ID
```

### 3. Install dependencies
```bash
uv sync
```

### 4. Run the app
```bash
export GOOGLE_CLOUD_PROJECT=YOUR-PROJECT-ID
uv run uvicorn app.main:app --reload
```

### 5. Open in your browser
```
http://localhost:8000/static/index.html
```

Other endpoints:
- `http://localhost:8000/health` — health check
- `http://localhost:8000/` — API status

---

## Run Tests

Unit tests cover the SQL-safety validation, dataset loading, and dashboard filter logic — the parts of the app that don't require a live Gemini call:
```bash
uv run pytest
```

For LLM-dependent behavior (scope enforcement, safety protocol, SQL-agent answer quality), see **Run Evaluations** below.

---

## Run Evaluations

The eval harness runs 40 test cases across three evaluation types:

| Type | Count | Method |
|---|---|---|
| Deterministic | 20 | Keyword/regex checks |
| Golden-reference MaaJ | 10 | LLM judge vs reference answer |
| Rubric MaaJ | 10 | LLM judge vs scoring rubric |

### Against local server
```bash
export GOOGLE_CLOUD_PROJECT=YOUR-PROJECT-ID
uv run python eval.py
```

### Against deployed Cloud Run
```bash
uv run python eval.py --url https://hr-advisor-36222488155.us-central1.run.app/chat
```

---

## Deploy to Cloud Run

### 1. Authenticate and set project
```bash
gcloud auth login
gcloud config set project YOUR-PROJECT-ID
```

### 2. Enable required APIs
```bash
gcloud services enable cloudbuild.googleapis.com run.googleapis.com
```

### 3. Deploy
```bash
gcloud builds submit .
```

The `cloudbuild.yaml` handles building the Docker image, pushing to GCR, and deploying to Cloud Run automatically.

---

## Prompt Design

The system prompt uses the **RISEN** framework (Role, Instructions, Steps, End Goal, Narrowing) with:
- 4 few-shot examples covering in-domain, out-of-scope, safety, and legal edge cases
- 3 named out-of-scope categories with positive framing
- A Python backstop post-generation classifier for safety and scope enforcement
