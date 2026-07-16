import json
import logging
import re
import sqlite3

from google.genai import types

logger = logging.getLogger(__name__)

SCHEMA_DESCRIPTION = """
Table: employees (one row per employee, from a real HR dataset of a company)
Columns:
  employee_id            INTEGER  unique employee id
  employee_name          TEXT     stored as 'Last, First' (e.g. 'Dunn, Amy')
  employee_display_name  TEXT     the same person's name as 'First Last' (e.g. 'Amy Dunn') --
                                   use this column, not employee_name, when joining against manager_name
  department              TEXT     e.g. 'Production', 'IT/IS', 'Software Engineering', 'Admin Offices', 'Sales', 'Executive Office'
  position                TEXT     job title
  state                   TEXT     US state code
  date_of_hire            TEXT     ISO date 'YYYY-MM-DD'
  date_of_termination     TEXT     ISO date 'YYYY-MM-DD', or NULL if still employed
  termd                   INTEGER  1 if terminated, 0 if still active
  term_reason             TEXT
  employment_status       TEXT     e.g. 'Active', 'Voluntarily Terminated', 'Terminated for Cause'
  manager_name            TEXT     the employee's manager, as 'First Last' (matches another row's employee_display_name)
  recruitment_source      TEXT     e.g. 'LinkedIn', 'Indeed', 'Referral', 'Diversity Job Fair'
  performance_score       TEXT     e.g. 'Exceeds', 'Fully Meets', 'Needs Improvement', 'PIP'
  engagement_survey       REAL     0-5 engagement score
  emp_satisfaction        INTEGER  0-5 satisfaction score
  salary                  INTEGER  annual salary in USD
  days_late_last_30       INTEGER
  absences                INTEGER
  gender                  TEXT     'M' or 'F'
  race                    TEXT     e.g. 'White', 'Black or African American', 'Asian', 'Two or more races',
                                    'American Indian or Alaska Native', 'Hispanic'
  marital_status          TEXT     e.g. 'Single', 'Married', 'Divorced', 'Widowed', 'Separated'
  date_of_birth           TEXT     ISO date 'YYYY-MM-DD' -- use for age, e.g. (julianday('now') - julianday(date_of_birth)) / 365.25
  special_projects_count  INTEGER  number of special projects the employee has worked on (workload signal)
  last_performance_review_date TEXT  ISO date 'YYYY-MM-DD' of the employee's most recent performance review

Org hierarchy: managers are themselves rows in this same table. To analyze a manager's team
(e.g. team size, team attrition rate), self-join the table:
  SELECT m.employee_name AS manager, COUNT(*) AS team_size, ...
  FROM employees e JOIN employees m ON e.manager_name = m.employee_display_name
  GROUP BY m.employee_name
Not every manager_name matches a row (a few are senior executives not tracked as individual
employees) -- a plain JOIN correctly excludes those, which is expected.
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

CHART_SYSTEM_INSTRUCTION = """You are a data visualization assistant. You are given a user's question, the SQL
that was run, and the resulting rows (a list of JSON objects, all sharing the same keys). Decide how to best
visualize this result and write a short narration.

Output ONLY a single JSON object wrapped in a ```json code fence, with exactly these keys:
{
  "chart_type": "bar" | "line" | "pie",
  "title": "short chart title",
  "x_field": "<one of the row keys, used as labels/x-axis>",
  "y_field": "<one of the row keys, used as the numeric value/y-axis>",
  "narration": "one or two sentence insight about the result, written for an HR analyst"
}

Rules:
- x_field and y_field MUST be exact key names taken from the given rows.
- y_field MUST be a numeric column (a count, rate, average, sum, etc.) -- never a text/category column.
- x_field and y_field MUST be different keys. If the rows only contain one numeric column and one
  category column, x_field is the category and y_field is the number.
- Use "line" only if x_field looks like a date or time period. Use "pie" only if there are 6 or fewer rows.
  Otherwise use "bar".
- Do not include any text outside the JSON code fence.
"""

SQL_FENCE_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.IGNORECASE | re.DOTALL)
ALLOWED_CHART_TYPES = {"bar", "line", "pie"}
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


def describe_chart(client, question: str, sql: str, rows: list[dict]) -> dict | None:
    prompt = f"User question: {question}\nSQL run: {sql}\nResult rows: {json.dumps(rows)}"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=CHART_SYSTEM_INSTRUCTION,
            temperature=0,
        ),
        contents=prompt,
    )
    match = JSON_FENCE_RE.search(response.text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def handle_dashboard_question(client, conn: sqlite3.Connection, question: str) -> dict:
    """Free-text question -> chart spec, for the dashboard's ask-a-question box.
    Always returns a dict with at least {"chart_type", "narration"}; falls back to
    chart_type="text" (no chart) on any unsupported/unsafe/invalid outcome."""
    text_fallback = {"chart_type": "text", "narration": FALLBACK_MESSAGE}
    try:
        sql = generate_sql(client, question)
        if not sql or not is_safe_select(sql) or "unsupported" in sql.lower():
            logger.info("Dashboard ask: unsupported or unsafe SQL for question=%r sql=%r", question, sql)
            return text_fallback

        rows = run_query(conn, sql)
        if not rows:
            return text_fallback

        chart = describe_chart(client, question, sql, rows)
        if not chart:
            logger.info("Dashboard ask: no chart spec returned for question=%r", question)
            return text_fallback

        chart_type = chart.get("chart_type")
        x_field = chart.get("x_field")
        y_field = chart.get("y_field")
        y_is_numeric = y_field in rows[0] and all(
            isinstance(row[y_field], (int, float)) for row in rows
        )
        if (
            chart_type not in ALLOWED_CHART_TYPES
            or x_field not in rows[0]
            or x_field == y_field
            or not y_is_numeric
        ):
            logger.info("Dashboard ask: invalid chart spec %r for question=%r", chart, question)
            return text_fallback

        return {
            "chart_type": chart_type,
            "title": chart.get("title") or question,
            "labels": [row[x_field] for row in rows],
            "values": [row[y_field] for row in rows],
            "narration": chart.get("narration", ""),
        }
    except Exception:
        logger.exception("Dashboard ask failed for question=%r", question)
        return text_fallback
