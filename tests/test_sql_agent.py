import sqlite3

import pytest

from app.sql_agent import extract_sql, is_safe_select, run_query


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
