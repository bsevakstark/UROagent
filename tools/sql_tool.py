"""
tools/sql_tool.py
-----------------
LangChain Tool — queries the mock SQLite EHR database.

The LLM generates a SQL SELECT statement; this tool executes it safely
and returns formatted results.  Only SELECT statements are permitted to
prevent any accidental writes to the clinical record store.
"""

import sqlite3
import re
from langchain.tools import tool

DB_PATH = "data/patient_records.db"

SCHEMA_DESCRIPTION = """
Table: patients
Columns:
  patient_id          INTEGER  – unique patient identifier
  age                 INTEGER  – patient age in years
  diagnosis           TEXT     – e.g. 'Prostate Cancer', 'Renal Cell Carcinoma',
                                 'Benign Prostatic Hyperplasia', 'Bladder Cancer',
                                 'Testicular Cancer'
  psa_level           REAL     – PSA in ng/mL (relevant for prostate conditions)
  tumor_stage         TEXT     – TNM stage string, e.g. 'T2a', 'T3b', or 'None'
  biopsy_gleason_score TEXT    – e.g. '3+4=7', '4+4=8', or 'N/A'
  treatment           TEXT     – current/planned treatment
  follow_up_months    INTEGER  – months since treatment started
"""


@tool
def query_ehr_database(sql_query: str) -> str:
    """
    Execute a SQL SELECT query against the urology EHR database and return the results.

    Use this tool when the question requires specific patient data such as:
    - Retrieving patients by diagnosis, PSA level, tumor stage, or age
    - Counting or aggregating patient records
    - Comparing clinical metrics across patient cohorts

    The tool accepts a raw SQL SELECT statement.  Do NOT use for INSERT/UPDATE/DELETE.

    Database schema:
    {schema}

    Example queries:
      SELECT * FROM patients WHERE diagnosis = 'Prostate Cancer' AND psa_level > 10;
      SELECT AVG(psa_level) FROM patients WHERE diagnosis = 'Prostate Cancer';
      SELECT patient_id, age, tumor_stage FROM patients WHERE tumor_stage LIKE 'T3%';
    """.format(
        schema=SCHEMA_DESCRIPTION
    )

    cleaned = sql_query.strip().upper()
    if not cleaned.startswith("SELECT"):
        return (
            "❌ Security Error: Only SELECT statements are permitted. "
            f"Received: {sql_query[:80]}"
        )

    sql_query = re.sub(r"```(?:sql)?|```", "", sql_query).strip()

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No records found matching the query criteria."

        columns = rows[0].keys()
        header = " | ".join(columns)
        separator = "-" * len(header)
        lines = [header, separator]
        for row in rows:
            lines.append(" | ".join(str(row[col]) for col in columns))

        result = "\n".join(lines)
        return f"Query returned {len(rows)} record(s):\n\n{result}"

    except sqlite3.Error as e:
        return f"❌ Database Error: {str(e)}\nQuery attempted: {sql_query}"
