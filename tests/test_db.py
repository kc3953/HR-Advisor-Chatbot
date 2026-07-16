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


# --------------------------------------------------------------------------- #
# Manager self-join (employee_display_name)
#
# Regression coverage: employee_name is stored "Last, First" while manager_name
# is stored "First Last" -- joining on those columns directly returns zero rows.
# employee_display_name exists specifically to fix that; these tests make sure
# it keeps working and don't let the join silently degrade back to zero matches.
# --------------------------------------------------------------------------- #

def test_employee_display_name_is_first_last_order():
    conn = get_connection()
    row = conn.execute(
        "SELECT employee_name, employee_display_name FROM employees "
        "WHERE employee_name = 'Dunn, Amy'"
    ).fetchone()
    assert row is not None
    assert row["employee_display_name"] == "Amy Dunn"


def test_naive_join_on_employee_name_returns_zero_rows():
    """Documents the bug this column exists to fix: without normalization, the
    join finds no matches at all because the two name columns use different
    orderings."""
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM employees e JOIN employees m ON e.manager_name = m.employee_name"
    ).fetchone()[0]
    assert count == 0


def test_normalized_join_covers_expected_employees():
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT m.employee_name AS manager, COUNT(*) AS team_size
        FROM employees e JOIN employees m ON e.manager_name = m.employee_display_name
        GROUP BY m.employee_name
        """
    ).fetchall()

    managers = [row["manager"] for row in rows]
    assert len(managers) == len(set(managers))  # no duplicate manager rows
    assert 0 < sum(row["team_size"] for row in rows) < 311  # covers most, not all, employees


def test_manager_attrition_rates_are_valid_percentages():
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT ROUND(100.0 * SUM(e.termd) / COUNT(*), 1) AS team_attrition_rate
        FROM employees e JOIN employees m ON e.manager_name = m.employee_display_name
        GROUP BY m.employee_name
        """
    ).fetchall()
    assert rows  # sanity: the join actually produced groups
    assert all(0 <= row["team_attrition_rate"] <= 100 for row in rows)
