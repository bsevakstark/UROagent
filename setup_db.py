"""
setup_db.py
-----------
Creates and populates a mock SQLite EHR database for UroAgent.
Run this once before launching the agent: `python setup_db.py`
"""

import sqlite3
import os


def create_mock_database():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/patient_records.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id   INTEGER PRIMARY KEY,
            age          INTEGER,
            diagnosis    TEXT,
            psa_level    REAL,
            tumor_stage  TEXT,
            biopsy_gleason_score TEXT,
            treatment    TEXT,
            follow_up_months INTEGER
        )
    """)

    mock_data = [
        (1,  65, "Prostate Cancer",              8.4,  "T2a",  "3+4=7",  "Radiation Therapy",          12),
        (2,  71, "Benign Prostatic Hyperplasia",  3.1,  "None", "N/A",    "Alpha-blockers",              6),
        (3,  58, "Prostate Cancer",              12.5,  "T3b",  "4+4=8",  "Radical Prostatectomy",      24),
        (4,  62, "Renal Cell Carcinoma",          1.2,  "T1a",  "N/A",    "Partial Nephrectomy",        18),
        (5,  77, "Prostate Cancer",              22.1,  "T4",   "5+5=10", "Androgen Deprivation Therapy", 3),
        (6,  54, "Bladder Cancer",                0.8,  "T2",   "N/A",    "TURBT + Intravesical Chemo",  9),
        (7,  69, "Prostate Cancer",               6.1,  "T1c",  "3+3=6",  "Active Surveillance",        36),
        (8,  80, "Benign Prostatic Hyperplasia",  4.7,  "None", "N/A",    "TURP",                        4),
        (9,  47, "Testicular Cancer",             0.5,  "T1",   "N/A",    "Orchiectomy",                18),
        (10, 73, "Prostate Cancer",              15.8,  "T3a",  "4+3=7",  "Brachytherapy",               6),
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO patients VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        mock_data,
    )
    conn.commit()
    conn.close()
    print("✅ Mock EHR database created at data/patient_records.db")
    print(f"   {len(mock_data)} patient records inserted.")


if __name__ == "__main__":
    create_mock_database()
