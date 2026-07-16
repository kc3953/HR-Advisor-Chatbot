import calendar
import os
import logging
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app import db, sql_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 1. Setup CORS and Static Files (Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves the frontend at the root URL
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Rate limiting — this service is public and unauthenticated, and every /chat or
# /api/dashboard/ask call spends real Vertex AI budget, so limits are keyed by
# client IP. Disable via RATE_LIMIT_ENABLED=false for local eval runs (eval.py
# fires ~40 sequential requests, which would otherwise trip the /chat limit).
limiter = Limiter(key_func=get_remote_address)
limiter.enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() != "false"
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 2. Vertex AI Client Setup
# In Cloud Run, authentication is automatic via the service account.
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    logger.error("GOOGLE_CLOUD_PROJECT environment variable is not set")
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set")

LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

logger.info(f"Initializing Vertex AI client with project: {PROJECT_ID}, location: {LOCATION}")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# 3. Define the System Prompt
# Prompt organization method: RISEN (Role, Instructions, Steps, End Goal, Narrowing)
SYSTEM_INSTRUCTION = """
## ROLE
You are a senior HR Advisor specializing in People Analytics. You assist employees and HR
professionals with questions about HR policies, workplace guidelines, employee benefits,
performance management, and people analytics concepts.

## INSTRUCTIONS
When answering a question, follow these steps in order:
1. Identify whether the question is in-domain (HR/people analytics), out-of-scope, or a safety situation.
2. If in-domain: provide a structured, bulleted answer with bold headers.
3. If out-of-scope: trigger the escape hatch response exactly as specified below.
4. If a safety or distress signal is detected: trigger the safety protocol immediately before anything else.
5. If legal advice is requested: redirect to legal counsel, then optionally provide general context only.

## END GOAL
Help users understand HR concepts, policies, and people analytics clearly and safely —
while protecting them from harm and keeping all answers within the HR domain.

## NARROWING (Positive Constraints)
- Only answer questions related to HR, people analytics, workplace policies, employee benefits,
  performance management, talent acquisition, workforce planning, and employee relations.
- Provide answers in a concise, bulleted format with bold headers.
- When citing policies or data, be clear about what is general best practice vs. company-specific.
- Always maintain a professional, empathetic, and neutral tone.

SAFETY PROTOCOL — DISTRESSED USERS:
- If a user expresses distress, mentions harassment, discrimination, mental health struggles,
  thoughts of self-harm, or any crisis situation, immediately respond with empathy and provide
  the following resources:
    * Employee Assistance Program (EAP): contact your HR department for your company's EAP number
    * Crisis Text Line: Text HOME to 741741
    * National Suicide Prevention Lifeline: 988
  Then advise them to speak directly with a human HR representative or manager.
- Never attempt to resolve legal complaints, harassment claims, or mental health crises yourself.
  Always escalate these to a qualified human professional.

SCOPE BOUNDARIES (Out-of-Scope Categories):
The following categories are outside your scope. Respond to each with the exact phrase shown:

1. TECHNOLOGY & CODING — questions about programming, software, IT systems, or technical tools
   that are unrelated to HR software (e.g. "write me a script", "how does TCP/IP work"):
   → "I am sorry, but I can only answer questions related to HR and people analytics. For other topics, please consult the appropriate resource."

2. PERSONAL LIFESTYLE — questions about food, travel, entertainment, sports, hobbies, or general life advice
   (e.g. "best restaurants in NYC", "who won the game"):
   → "I am sorry, but I can only answer questions related to HR and people analytics. For other topics, please consult the appropriate resource."

3. FINANCIAL & LEGAL ADVICE — questions requiring specific legal rulings, investment advice, tax guidance,
   or any professional opinion outside HR scope (e.g. "should I buy this stock", "how do I file my taxes"):
   → "This question may require legal or financial expertise. Please consult your company's legal counsel, an employment attorney, or a licensed financial advisor."

EXAMPLES (follow this format and tone exactly):

Example 1 — In-domain, people analytics concept:
User: What is employee attrition rate and how is it calculated?
Assistant:
- **Definition:** Employee attrition rate measures the percentage of employees who leave an organization over a given period.
- **Formula:** (Number of employees who left ÷ Average headcount) × 100
- **Example:** If 10 employees left out of an average of 200, the attrition rate is 5%.
- **Note:** This is a general best practice formula; your company may track this differently in its HRIS.

Example 2 — Out-of-scope, escape hatch:
User: Can you help me write a Python script?
Assistant: I am sorry, but I can only answer questions related to HR and people analytics. For other topics, please consult the appropriate resource.

Example 3 — Safety protocol, distressed user:
User: My manager has been bullying me for months and I feel like I can't take it anymore.
Assistant: I'm really sorry to hear you're going through this — what you're feeling is valid, and you deserve support. Please reach out to the following resources right away:
- **Your HR department** — report the situation to a human HR representative who can take formal action.
- **Employee Assistance Program (EAP)** — contact your HR team for your company's confidential EAP number.
- **Crisis Text Line** — Text HOME to 741741 for free, confidential support 24/7.
- **National Suicide Prevention Lifeline** — call or text 988 if you are in crisis.
You do not have to handle this alone. Please speak with a qualified HR professional or counselor directly.

Example 4 — Uncertainty / legal boundary:
User: Can I sue my company for wrongful termination?
Assistant: This question may require legal expertise. Please consult your company's legal counsel or an employment attorney. What I can share is general context:
- **Wrongful termination** typically refers to being fired in violation of employment law or a contract.
- Common grounds include discrimination, retaliation, or breach of contract.
- An employment attorney can assess whether your specific situation warrants legal action.
"""

# 4. Python Backstop — post-generation safety classifier
# Catches cases where the LLM fails to apply the system prompt correctly.

DISTRESS_KEYWORDS = [
    "harass", "bully", "bullying", "assault", "abuse", "discriminat",
    "hopeless", "worthless", "don't want to be here", "can't take it",
    "end it all", "hurt myself", "self-harm", "suicid", "overwhelmed and",
    "breaking down", "mental breakdown",
]

OUT_OF_SCOPE_KEYWORDS = [
    "recipe", "cook", "restaurant", "netflix", "movie", "song", "sport",
    "football", "basketball", "nfl", "nba", "super bowl", "tourist",
    "vacation", "travel", "stock market", "crypto", "bitcoin", "investment",
    "write a script", "python script", "javascript", "html code", "hack",
    "write code", "debug", "tcp/ip", "machine learning model",
]

# Questions matching these keywords are routed to the text-to-SQL agent instead
# of the conceptual RISEN prompt, so they get answered with real computed numbers
# from the HR dataset rather than a generic explanation.
DATA_QUERY_KEYWORDS = [
    "attrition rate", "attrition by", "headcount", "how many employees",
    "how many people", "turnover", "hiring funnel", "recruitment source",
    "hires by", "hired in", "time to fill", "tenure", "average salary",
    "average tenure", "engagement score", "satisfaction score",
    "performance score", "by department", "trend over time",
]

# Even if a message matches a DATA_QUERY_KEYWORDS term, these markers indicate
# the user is asking for a definition/explanation (e.g. "what IS attrition rate
# and how is it CALCULATED"), not a computed value — keep those on the
# conceptual RISEN path instead of routing to the SQL agent.
CONCEPTUAL_MARKERS = [
    "how is it calculated", "how is that calculated", "how are they calculated",
    "how does", "difference between", "explain", "define", "definition of",
    "what does it mean", "concept of",
]

DISTRESS_FALLBACK = """I'm really sorry to hear you're going through a difficult time. What you're feeling matters, and you deserve support.

Please reach out to the following resources right away:
- **Your HR department** — speak with a human HR representative who can take formal action.
- **Employee Assistance Program (EAP)** — contact your HR team for your company's confidential EAP number.
- **Crisis Text Line** — Text HOME to 741741 for free, confidential support 24/7.
- **National Suicide Prevention Lifeline** — call or text **988** if you are in crisis.

You do not have to handle this alone. Please speak with a qualified HR professional or counselor directly."""

OUT_OF_SCOPE_FALLBACK = "I am sorry, but I can only answer questions related to HR and people analytics. For other topics, please consult the appropriate resource."


def backstop_classifier(user_message: str, llm_response: str) -> str | None:
    """
    Post-generation backstop. Returns an override response if the LLM
    failed to handle a distress signal or out-of-scope question correctly.
    Returns None if the LLM response looks correct.
    """
    msg_lower = user_message.lower()

    # Check 1: Distress signal in user message — ensure crisis resources are present
    if any(kw in msg_lower for kw in DISTRESS_KEYWORDS):
        if not any(marker in llm_response for marker in ["988", "741741", "EAP"]):
            logger.warning("Backstop triggered: distress signal detected, LLM missed safety protocol")
            return DISTRESS_FALLBACK

    # Check 2: Out-of-scope topic — ensure LLM refused correctly
    if any(kw in msg_lower for kw in OUT_OF_SCOPE_KEYWORDS):
        refusal_markers = ["only answer questions", "people analytics", "appropriate resource"]
        if not any(marker in llm_response.lower() for marker in refusal_markers):
            logger.warning("Backstop triggered: out-of-scope topic detected, LLM did not refuse")
            return OUT_OF_SCOPE_FALLBACK

    return None


class ChatRequest(BaseModel):
    message: str

@app.get("/favicon.ico", status_code=204)
async def favicon():
    pass

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "service": "domain-chatbot"}

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, payload: ChatRequest):
    logger.info(f"Received chat request: {payload.message[:100]}...")
    msg_lower = payload.message.lower()
    try:
        # Route quantitative questions to the text-to-SQL agent, which answers
        # with real numbers computed from the HR dataset. Safety and out-of-scope
        # signals still take priority and fall through to the standard path below,
        # which already handles them (and is backed up by backstop_classifier).
        is_distress = any(kw in msg_lower for kw in DISTRESS_KEYWORDS)
        is_out_of_scope = any(kw in msg_lower for kw in OUT_OF_SCOPE_KEYWORDS)
        is_data_query = any(kw in msg_lower for kw in DATA_QUERY_KEYWORDS)
        is_conceptual = any(kw in msg_lower for kw in CONCEPTUAL_MARKERS)

        if is_data_query and not is_distress and not is_out_of_scope and not is_conceptual:
            logger.info("Routing to SQL agent")
            conn = db.get_connection()
            data_response = sql_agent.handle_data_question(client, conn, payload.message)
            return {"response": data_response}

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3, # Low temp for deterministic factual answers
            ),
            contents=payload.message
        )
        llm_response = response.text

        # Run Python backstop to catch LLM misses
        override = backstop_classifier(payload.message, llm_response)
        if override:
            return {"response": override}

        logger.info(f"Generated response successfully")
        return {"response": llm_response}
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal AI Error")

@app.get("/")
async def root():
    return {"message": "Chatbot API is running. Go to /static/index.html to get started."}


# 5. Dashboard API — pure SQL against the HR dataset, no LLM calls (except
# /api/dashboard/ask), so the core dashboard loads fast and independent of
# Vertex AI availability.

def _filter_clause(
    department: str | None,
    recruitment_source: str | None,
    date_from: str | None,
    date_to: str | None,
    table_alias: str = "",
) -> tuple[str, list]:
    """Builds a parameterized WHERE fragment (e.g. " AND department = ?") shared
    by all filterable dashboard endpoints. Always uses placeholders, never string
    interpolation, even though values currently only come from UI dropdowns.

    table_alias qualifies column names (e.g. "e" -> "e.department = ?") -- required
    for self-joins like /api/dashboard/manager-teams, where an unqualified column
    name would be ambiguous between the two joined instances of `employees`."""
    prefix = f"{table_alias}." if table_alias else ""
    clauses = []
    params: list = []
    if department:
        clauses.append(f"{prefix}department = ?")
        params.append(department)
    if recruitment_source:
        clauses.append(f"{prefix}recruitment_source = ?")
        params.append(recruitment_source)
    if date_from:
        clauses.append(f"{prefix}date_of_hire >= ?")
        params.append(date_from)
    if date_to:
        clauses.append(f"{prefix}date_of_hire <= ?")
        params.append(date_to)
    clause = (" AND " + " AND ".join(clauses)) if clauses else ""
    return clause, params


@app.get("/api/dashboard/filter-options")
@limiter.limit("60/minute")
async def dashboard_filter_options(request: Request):
    """Distinct filter values and hire-date bounds, to drive the filter UI dynamically."""
    conn = db.get_connection()
    departments = [r[0] for r in conn.execute("SELECT DISTINCT department FROM employees ORDER BY department")]
    sources = [r[0] for r in conn.execute("SELECT DISTINCT recruitment_source FROM employees ORDER BY recruitment_source")]
    bounds = conn.execute("SELECT MIN(date_of_hire), MAX(date_of_hire) FROM employees").fetchone()
    return {
        "departments": departments,
        "recruitment_sources": sources,
        "date_of_hire_min": bounds[0],
        "date_of_hire_max": bounds[1],
    }


@app.get("/api/dashboard/headcount")
@limiter.limit("60/minute")
async def dashboard_headcount(
    request: Request,
    department: str | None = Query(None),
    recruitment_source: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    """Active headcount at the end of each month, from real hire/termination dates."""
    conn = db.get_connection()
    where_clause, params = _filter_clause(department, recruitment_source, date_from, date_to)

    bounds = conn.execute(
        f"""
        SELECT MIN(date_of_hire), COALESCE(MAX(date_of_termination), MAX(date_of_hire))
        FROM employees WHERE 1=1 {where_clause}
        """,
        params,
    ).fetchone()

    if not bounds[0]:
        return {"labels": [], "headcount": []}

    start, end = bounds[0][:7], bounds[1][:7]  # 'YYYY-MM'

    months = []
    y, m = (int(x) for x in start.split("-"))
    end_y, end_m = (int(x) for x in end.split("-"))
    while (y, m) <= (end_y, end_m):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1

    counts = []
    for month in months:
        y, m = (int(x) for x in month.split("-"))
        last_day = calendar.monthrange(y, m)[1]
        month_end = f"{month}-{last_day:02d}"
        row = conn.execute(
            f"""
            SELECT COUNT(*) FROM employees
            WHERE date_of_hire <= ?
              AND (date_of_termination IS NULL OR date_of_termination > ?)
              {where_clause}
            """,
            [month_end, month_end] + params,
        ).fetchone()
        counts.append(row[0])

    return {"labels": months, "headcount": counts}


@app.get("/api/dashboard/attrition")
@limiter.limit("60/minute")
async def dashboard_attrition(
    request: Request,
    department: str | None = Query(None),
    recruitment_source: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    """Attrition rate by department: terminated employees / total employees ever in that department."""
    conn = db.get_connection()
    where_clause, params = _filter_clause(department, recruitment_source, date_from, date_to)
    rows = conn.execute(
        f"""
        SELECT department,
               SUM(termd) AS terminated,
               COUNT(*) AS total,
               ROUND(100.0 * SUM(termd) / COUNT(*), 1) AS attrition_rate
        FROM employees
        WHERE 1=1 {where_clause}
        GROUP BY department
        ORDER BY attrition_rate DESC
        """,
        params,
    ).fetchall()
    return {"departments": [dict(r) for r in rows]}


@app.get("/api/dashboard/recruitment-source")
@limiter.limit("60/minute")
async def dashboard_recruitment_source(
    request: Request,
    department: str | None = Query(None),
    recruitment_source: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    """Hire counts by recruitment source (real-data stand-in for a hiring funnel)."""
    conn = db.get_connection()
    where_clause, params = _filter_clause(department, recruitment_source, date_from, date_to)
    rows = conn.execute(
        f"""
        SELECT recruitment_source, COUNT(*) AS hires
        FROM employees
        WHERE 1=1 {where_clause}
        GROUP BY recruitment_source
        ORDER BY hires DESC
        """,
        params,
    ).fetchall()
    return {"sources": [dict(r) for r in rows]}


@app.get("/api/dashboard/manager-teams")
@limiter.limit("60/minute")
async def dashboard_manager_teams(
    request: Request,
    department: str | None = Query(None),
    recruitment_source: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    """Team size and attrition rate per manager -- a self-join of employees to
    itself on manager_name = employee_display_name (see app/db.py: the two name
    columns use different orderings in the source data and must be normalized
    before they can be joined)."""
    conn = db.get_connection()
    where_clause, params = _filter_clause(department, recruitment_source, date_from, date_to, table_alias="e")
    rows = conn.execute(
        f"""
        SELECT m.employee_name AS manager,
               COUNT(*) AS team_size,
               ROUND(100.0 * SUM(e.termd) / COUNT(*), 1) AS team_attrition_rate,
               ROUND(AVG(e.engagement_survey), 2) AS team_avg_engagement
        FROM employees e
        JOIN employees m ON e.manager_name = m.employee_display_name
        WHERE 1=1 {where_clause}
        GROUP BY m.employee_name
        ORDER BY team_size DESC
        """,
        params,
    ).fetchall()
    return {"managers": [dict(r) for r in rows]}


class DashboardAskRequest(BaseModel):
    question: str


@app.post("/api/dashboard/ask")
@limiter.limit("10/minute")
async def dashboard_ask(request: Request, payload: DashboardAskRequest):
    """Free-text question -> a chart, generated live via the text-to-SQL agent."""
    conn = db.get_connection()
    result = sql_agent.handle_dashboard_question(client, conn, payload.question)
    return result
