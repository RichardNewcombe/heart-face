"""
Setup the Life Planner doc_sync_queue database.
Creates the SQLite database and doc_sync_queue table if they don't exist.
"""

import sqlite3
import os

DB_PATH = os.environ.get("DATABASE_URL", "/home/ubuntu/life_planner.db")
# Strip sqlite:/// prefix if present
if DB_PATH.startswith("sqlite:///"):
    DB_PATH = DB_PATH[len("sqlite:///"):]

def setup():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_sync_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL CHECK(action IN ('add_row', 'strikethrough', 'remove_strikethrough')),
            table_name TEXT NOT NULL DEFAULT 'schedule',
            row_data TEXT,          -- JSON: for add_row, contains cell values
            match_text TEXT,        -- For strikethrough/remove_strikethrough: text to find in table
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'done', 'failed')),
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP
        )
    """)

    conn.commit()
    print(f"Database initialized at {DB_PATH}")
    print(f"Table doc_sync_queue ready.")

    # Show current queue status
    cur.execute("SELECT status, COUNT(*) FROM doc_sync_queue GROUP BY status")
    rows = cur.fetchall()
    if rows:
        print("\nCurrent queue status:")
        for status, count in rows:
            print(f"  {status}: {count}")
    else:
        print("\nQueue is empty.")

    conn.close()

if __name__ == "__main__":
    setup()
