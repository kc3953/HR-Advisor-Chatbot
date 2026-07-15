import logging
import re
import sqlite3

from google.genai import types

logger = logging.getLogger(__name__)

SCHEMA_DESCRIPTION = """
Table: employees (one row per employee, from a real HR dataset of a company)
Columns:
  employee_id          INTEGER  unique employee id
  employee_name        TEXT
  department            TEXT     e.g. 'Production', 'IT/IS', 'Software Engineering', 'Admin Offices', 'Sales', 'Executive Office'
  position              TEXT     job title
  state                 TEXT     US state code
  date_of_hire          TEXT     ISO date 'YYYY-MM-DD'
  date_of_termination   TEXT     ISO date 'YYYY-MM-DD', or NULL if still employed
  termd                 INTEGER  1 if terminated, 0 if still active
  term_reason           TEXT
  employment_status     TEXT     e.g. 'Active', 'Voluntarily Terminated', 'Terminated for Cause'
  manager_name          TEXT
  recruitment_source    TEXT     e.g. 'LinkedIn', 'Indeed', 'Referral', 'Diversity Job Fair'
  performance_score     TEXT     e.g. 'Exceeds', 'Fully Meets', 'Needs Improvement', 'PIP'
  engagement_survey     REAL     0-5 engagement score
  emp_satisfaction      INTEGER  0-5 satisfaction score
  salary                INTEGER  annual salary in USD
  days_late_last_30     INTEGER
  absences              INTEGER
"""

SQL_SYSTEM_INSTRUCTION = f"""You are a SQL generator for a SQLite database of HR/people-analytics data.

{SCHEMA_DESCRIPTION}

Rules:
- Output ONLY a single read-only SELECT statement, wrapped in a ```sql code fence. No prose, no explanation.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, or PRAGMA.
- Never use more than one statement (no semicolons except an optional trailing one).
- Use SQLite date functions (strftime, julianday) for date math against date_of_hire / date_of_termination.
- If the question cannot be answered from this schema, output: ```sql\nSELECT 'unsupported' AS error\n```
"""

NARRATION_SYSTEM_INSTRUCTION = """You are a senior People Analytics advisor. You were given a user's question, the
SQL query that was run against the company's HR database, and the resulting rows. Turn this into a clear,
concise, bulleted answer with bold headers, in this shape:
- **Answer:** the headline number(s), directly answering the question.
- **Insight:** one sentence of context on what the number means.
- **Recommendation:** one sentence suggesting a next step or thing to watch, if relevant (omit if not applicable).
Keep it short. Do not mention SQL or the database in your answer — speak as an HR analyst presenting findings.
"""

SQL_FENCE_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|DETACH|PRAGMA|CREATE|REPLACE|VACUUM)\b",
    re.IGNORECASE,
)

FALLBACK_MESSAGE = (
    "I wasn't able to compute an answer to that from the HR dataset. "
    "Try rephrasing your question, e.g. \"What is the attrition rate by department?\" "
    "or \"How many employees were hired in 2015?\""
)


def extract_sql(llm_output: str) -> str | None:
    match = SQL_FENCE_RE.search(llm_output)
    if not match:
        return None
    return match.group(1).strip()


def is_safe_select(sql: str) -> bool:
    if not sql:
        return False
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        return False
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        return False
    if FORBIDDEN_KEYWORDS.search(stripped):
        return False
    return True


def run_query(conn: sqlite3.Connection, sql: str) -> list[dict]:
    clean_sql = sql.strip().rstrip(";")
    if not re.search(r"\bLIMIT\b", clean_sql, re.IGNORECASE):
        clean_sql += " LIMIT 200"
    rows = conn.execute(clean_sql).fetchall()
    return [dict(row) for row in rows]


def generate_sql(client, question: str) -> str | None:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SQL_SYSTEM_INSTRUCTION,
            temperature=0,
        ),
        contents=question,
    )
    return extract_sql(response.text or "")


def narrate_result(client, question: str, sql: str, rows: list[dict]) -> str:
    prompt = f"User question: {question}\nSQL run: {sql}\nResult rows: {rows}"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=NARRATION_SYSTEM_INSTRUCTION,
            temperature=0.3,
        ),
        contents=prompt,
    )
    return response.text


def handle_data_question(client, conn: sqlite3.Connection, question: str) -> str:
    try:
        sql = generate_sql(client, question)
        if not sql or not is_safe_select(sql) or "unsupported" in sql.lower():
            logger.info("SQL agent: unsupported or unsafe SQL for question=%r sql=%r", question, sql)
            return FALLBACK_MESSAGE

        rows = run_query(conn, sql)
        if not rows:
            return FALLBACK_MESSAGE

        return narrate_result(client, question, sql, rows)
    except Exception:
        logger.exception("SQL agent failed for question=%r", question)
        return FALLBACK_MESSAGE
