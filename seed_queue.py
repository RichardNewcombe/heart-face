"""
Seed the doc_sync_queue with sample items to demonstrate all three actions.
"""

import json
import os
import sqlite3

DB_PATH = os.environ.get("DATABASE_URL", "/home/ubuntu/life_planner.db")
if DB_PATH.startswith("sqlite:///"):
    DB_PATH = DB_PATH[len("sqlite:///"):]

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Check if there are already pending items
cur.execute("SELECT COUNT(*) FROM doc_sync_queue WHERE status = 'pending'")
pending = cur.fetchone()[0]
if pending > 0:
    print(f"Already {pending} pending items in queue. Skipping seed.")
    conn.close()
    exit(0)

items = [
    # 1. Add a new event row
    {
        "action": "add_row",
        "table_name": "schedule",
        "row_data": json.dumps({
            "dates": "12 Apr",
            "item": "Spring Trail Run",
            "location": "Portland, OR",
            "status": "Planned",
            "notes": "Auto-added by doc sync processor",
        }),
        "match_text": None,
    },
    # 2. Apply strikethrough to a completed event (Weekly Planning Review on 30 Mar is already struck through, so pick another)
    {
        "action": "strikethrough",
        "table_name": "schedule",
        "row_data": None,
        "match_text": "Life Planner Sync Verification",
    },
    # 3. Remove strikethrough from an event (Weekly Planning Review on 30 Mar has strikethrough)
    {
        "action": "remove_strikethrough",
        "table_name": "schedule",
        "row_data": None,
        "match_text": "Weekly Planning Review",
    },
]

for item in items:
    cur.execute(
        """INSERT INTO doc_sync_queue (action, table_name, row_data, match_text)
           VALUES (?, ?, ?, ?)""",
        (item["action"], item["table_name"], item["row_data"], item["match_text"])
    )

conn.commit()
print(f"Seeded {len(items)} items into doc_sync_queue.")

cur.execute("SELECT id, action, match_text, status FROM doc_sync_queue WHERE status = 'pending'")
for row in cur.fetchall():
    print(f"  #{row[0]}: {row[1]} | match={row[2]} | status={row[3]}")

conn.close()
