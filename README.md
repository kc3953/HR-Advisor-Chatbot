# HR Advisor — People Analytics Chatbot

**Team Member**: Chung-Yeh Yang(cy2816), Kuan-Ting Chen(kc3953)

A domain-specific chatbot that answers HR and people analytics questions, enforces strict scope boundaries, and handles distressed users safely. Built with FastAPI + Gemini on Vertex AI, deployed on Google Cloud Run.

[**Live URL Link**](https://hr-advisor-ixylabcy6a-uc.a.run.app/static/index.html)

## What It Does

- Answers questions about HR policies, employee benefits, performance management, talent acquisition, workforce planning, and people analytics concepts
- Refuses out-of-scope topics (technology/coding, lifestyle, financial/legal advice) with a clear explanation
- Detects distressed users and immediately provides crisis resources (988, Crisis Text Line, EAP)
- Uses a Python backstop classifier to catch cases where the LLM misses scope or safety rules

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.5 Flash via Vertex AI |
| Backend | FastAPI (Python 3.12) |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Google Cloud Run |
| Package manager | uv |

---

## Project Structure

```
domain-chatbot/
├── app/
│   ├── main.py          # FastAPI app, system prompt, backstop classifier
│   └── static/
│       └── index.html   # Chat UI
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
uv run python eval.py --url https://hr-advisor-ixylabcy6a-uc.a.run.app/chat
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
