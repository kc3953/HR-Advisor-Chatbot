import csv
import os
import sqlite3
import threading

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "HRDataset_v14.csv")

_connection: sqlite3.Connection | None = None
_lock = threading.Lock()


def _parse_date(value: str) -> str | None:
    """Convert M/D/YYYY (as used in the CSV) to ISO YYYY-MM-DD for SQLite date functions."""
    value = (value or "").strip()
    if not value:
        return None
    month, day, year = value.split("/")
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _load_dataset(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE employees (
            employee_id INTEGER,
            employee_name TEXT,
            department TEXT,
            position TEXT,
            state TEXT,
            date_of_hire TEXT,
            date_of_termination TEXT,
            termd INTEGER,
            term_reason TEXT,
            employment_status TEXT,
            manager_name TEXT,
            recruitment_source TEXT,
            performance_score TEXT,
            engagement_survey REAL,
            emp_satisfaction INTEGER,
            salary INTEGER,
            days_late_last_30 INTEGER,
            absences INTEGER
        )
    """)

    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            (
                int(row["EmpID"]),
                row["Employee_Name"].strip(),
                row["Department"].strip(),
                row["Position"].strip(),
                row["State"].strip(),
                _parse_date(row["DateofHire"]),
                _parse_date(row["DateofTermination"]),
                int(row["Termd"]),
                row["TermReason"].strip(),
                row["EmploymentStatus"].strip(),
                row["ManagerName"].strip(),
                row["RecruitmentSource"].strip(),
                row["PerformanceScore"].strip(),
                float(row["EngagementSurvey"]) if row["EngagementSurvey"] else None,
                int(row["EmpSatisfaction"]) if row["EmpSatisfaction"] else None,
                int(row["Salary"]) if row["Salary"] else None,
                int(row["DaysLateLast30"]) if row["DaysLateLast30"] else 0,
                int(row["Absences"]) if row["Absences"] else 0,
            )
            for row in reader
        ]

    conn.executemany(
        """
        INSERT INTO employees (
            employee_id, employee_name, department, position, state,
            date_of_hire, date_of_termination, termd, term_reason, employment_status,
            manager_name, recruitment_source, performance_score, engagement_survey,
            emp_satisfaction, salary, days_late_last_30, absences
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def get_connection() -> sqlite3.Connection:
    """Returns a lazily-initialized, process-wide in-memory SQLite connection
    seeded from the HRDataset_v14 CSV on first access."""
    global _connection
    if _connection is None:
        with _lock:
            if _connection is None:
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                conn.row_factory = sqlite3.Row
                _load_dataset(conn)
                _connection = conn
    return _connection
