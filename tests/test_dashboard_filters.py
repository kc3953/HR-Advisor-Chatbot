import sqlite3

from app.main import _filter_clause


def test_no_filters_returns_empty_clause():
    clause, params = _filter_clause(None, None, None, None)
    assert clause == ""
    assert params == []


def test_single_department_filter():
    clause, params = _filter_clause("Sales", None, None, None)
    assert clause == " AND department = ?"
    assert params == ["Sales"]


def test_all_filters_combined_in_order():
    clause, params = _filter_clause("Sales", "LinkedIn", "2015-01-01", "2018-01-01")
    assert clause == " AND department = ? AND recruitment_source = ? AND date_of_hire >= ? AND date_of_hire <= ?"
    assert params == ["Sales", "LinkedIn", "2015-01-01", "2018-01-01"]


def test_clause_produces_valid_executable_sql_when_appended_after_where_1_equals_1():
    clause, params = _filter_clause("Sales", "LinkedIn", "2015-01-01", "2018-01-01")
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE employees (department TEXT, recruitment_source TEXT, date_of_hire TEXT)")
    conn.execute(
        "INSERT INTO employees VALUES ('Sales', 'LinkedIn', '2016-01-01'), ('IT/IS', 'Indeed', '2016-01-01')"
    )
    sql = f"SELECT * FROM employees WHERE 1=1 {clause}"
    rows = conn.execute(sql, params).fetchall()
    assert rows == [("Sales", "LinkedIn", "2016-01-01")]


def test_empty_string_filters_are_treated_as_unset():
    clause, params = _filter_clause("", "", "", "")
    assert clause == ""
    assert params == []
