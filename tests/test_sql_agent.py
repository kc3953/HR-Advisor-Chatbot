import sqlite3

import pytest

from app.sql_agent import extract_sql, is_safe_select, run_query, handle_dashboard_question


# --------------------------------------------------------------------------- #
# is_safe_select
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sql", [
    "SELECT * FROM employees",
    "select department, count(*) from employees group by department",
    "  SELECT id FROM employees WHERE termd = 1",
    "SELECT created_at FROM employees",  # "CREATE" substring, not the keyword
    "SELECT id FROM employees;",  # single trailing semicolon is fine
])
def test_is_safe_select_accepts_valid_select(sql):
    assert is_safe_select(sql) is True


@pytest.mark.parametrize("sql", [
    None,
    "",
    "DROP TABLE employees",
    "INSERT INTO employees (id) VALUES (1)",
    "UPDATE employees SET salary = 0",
    "DELETE FROM employees",
    "ALTER TABLE employees ADD COLUMN x TEXT",
    "ATTACH DATABASE 'x.db' AS x",
    "PRAGMA table_info(employees)",
    "CREATE TABLE evil (id INTEGER)",
    "SELECT * FROM employees; DROP TABLE employees",
    "employees SELECT *",  # doesn't start with SELECT
])
def test_is_safe_select_rejects_unsafe_or_invalid(sql):
    assert is_safe_select(sql) is False


# --------------------------------------------------------------------------- #
# extract_sql
# --------------------------------------------------------------------------- #

def test_extract_sql_parses_fenced_block():
    llm_output = "```sql\nSELECT * FROM employees\n```"
    assert extract_sql(llm_output) == "SELECT * FROM employees"


def test_extract_sql_is_case_insensitive_on_fence_marker():
    llm_output = "```SQL\nSELECT 1\n```"
    assert extract_sql(llm_output) == "SELECT 1"


def test_extract_sql_returns_none_without_fence():
    assert extract_sql("SELECT * FROM employees") is None
    assert extract_sql("") is None


# --------------------------------------------------------------------------- #
# run_query
# --------------------------------------------------------------------------- #

@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE t (id INTEGER)")
    connection.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(300)])
    connection.commit()
    yield connection
    connection.close()


def test_run_query_adds_limit_when_missing(conn):
    rows = run_query(conn, "SELECT * FROM t")
    assert len(rows) == 200  # default LIMIT 200 injected


def test_run_query_preserves_existing_limit(conn):
    rows = run_query(conn, "SELECT * FROM t LIMIT 5")
    assert len(rows) == 5


def test_run_query_returns_list_of_dicts(conn):
    rows = run_query(conn, "SELECT * FROM t LIMIT 1")
    assert rows == [{"id": 0}]


# --------------------------------------------------------------------------- #
# handle_dashboard_question
#
# Regression coverage for a live bug: asking "which recruitment source has
# the most hires?" produced SQL selecting only the category column (no count),
# and the LLM's chart spec set x_field == y_field == "recruitment_source" --
# rendering a chart whose "values" were category names instead of numbers.
# --------------------------------------------------------------------------- #

class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate_content(self, **kwargs):
        return _FakeResponse(self._responses.pop(0))


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


@pytest.fixture
def source_conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE employees (recruitment_source TEXT)")
    connection.execute("INSERT INTO employees VALUES ('Indeed')")
    connection.commit()
    yield connection
    connection.close()


def test_rejects_chart_spec_with_non_numeric_y_field(source_conn):
    sql_response = "```sql\nSELECT recruitment_source FROM employees LIMIT 1\n```"
    chart_response = (
        '```json\n{"chart_type": "bar", "title": "Top Source", '
        '"x_field": "recruitment_source", "y_field": "recruitment_source", '
        '"narration": "n"}\n```'
    )
    client = _FakeClient([sql_response, chart_response])

    result = handle_dashboard_question(client, source_conn, "which source has the most hires?")
    assert result["chart_type"] == "text"


def test_accepts_chart_spec_with_numeric_y_field():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE employees (recruitment_source TEXT)")
    conn.executemany(
        "INSERT INTO employees VALUES (?)", [("Indeed",)] * 3 + [("LinkedIn",)] * 1
    )
    conn.commit()

    sql_response = (
        "```sql\nSELECT recruitment_source, COUNT(*) AS hires FROM employees "
        "GROUP BY recruitment_source ORDER BY hires DESC\n```"
    )
    chart_response = (
        '```json\n{"chart_type": "bar", "title": "Hires by Source", '
        '"x_field": "recruitment_source", "y_field": "hires", '
        '"narration": "Indeed leads."}\n```'
    )
    client = _FakeClient([sql_response, chart_response])

    result = handle_dashboard_question(client, conn, "hires by source?")
    assert result["chart_type"] == "bar"
    assert result["labels"] == ["Indeed", "LinkedIn"]
    assert result["values"] == [3, 1]
