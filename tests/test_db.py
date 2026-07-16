import re

from app.db import get_connection

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def test_loads_all_rows():
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    assert count == 311


def test_termd_is_boolean_flag():
    conn = get_connection()
    values = {row[0] for row in conn.execute("SELECT DISTINCT termd FROM employees")}
    assert values <= {0, 1}


def test_date_of_hire_is_iso_format():
    conn = get_connection()
    rows = conn.execute("SELECT date_of_hire FROM employees WHERE date_of_hire IS NOT NULL").fetchall()
    assert len(rows) == 311
    assert all(ISO_DATE_RE.match(row[0]) for row in rows)


def test_date_of_termination_is_iso_or_null():
    conn = get_connection()
    rows = conn.execute("SELECT date_of_termination FROM employees").fetchall()
    for (value,) in rows:
        assert value is None or ISO_DATE_RE.match(value)


def test_department_values_are_stripped():
    conn = get_connection()
    departments = [row[0] for row in conn.execute("SELECT DISTINCT department FROM employees")]
    assert all(d == d.strip() for d in departments)
    assert all(d for d in departments)  # none blank


def test_connection_is_cached_singleton():
    assert get_connection() is get_connection()
