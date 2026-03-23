#!/usr/bin/env python3
"""
Life Planner — Google Doc Sync Processor

Connects to the Life Planner database, reads pending items from `doc_sync_queue`,
and processes them using the `gws` CLI to modify the Google Doc.

Supported actions:
  - add_row:              Adds a new event row to the main schedule table
  - strikethrough:        Applies strikethrough formatting to completed events
  - remove_strikethrough: Removes strikethrough from uncompleted events

Items that fail are retried up to 3 times automatically.
If the gws CLI token is expired, the script fails gracefully — items remain
in the queue for the next run.
"""

import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOC_ID = "1QV8vFsycef2-QUCk6o44k-sVbk_bpTfm1ji7wYD_DXs"

DB_PATH = os.environ.get("DATABASE_URL", "/home/ubuntu/life_planner.db")
if DB_PATH.startswith("sqlite:///"):
    DB_PATH = DB_PATH[len("sqlite:///"):]

# The main schedule table is the 5-column table (Dates, Item, Location, Status, Notes)
SCHEDULE_TABLE_COLS = 5


# ---------------------------------------------------------------------------
# Helpers — gws CLI wrappers
# ---------------------------------------------------------------------------

def gws_get_doc():
    """Fetch the current document structure via gws CLI."""
    result = subprocess.run(
        ["gws", "docs", "documents", "get",
         "--params", json.dumps({"documentId": DOC_ID})],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"gws docs get failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def gws_batch_update(requests):
    """Send a batchUpdate to the document via gws CLI."""
    body = {"requests": requests}
    result = subprocess.run(
        ["gws", "docs", "documents", "batchUpdate",
         "--params", json.dumps({"documentId": DOC_ID}),
         "--json", json.dumps(body)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"gws batchUpdate failed: {result.stderr.strip()}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def check_gws_token():
    """Quick health-check: can we read the doc at all?"""
    try:
        doc = gws_get_doc()
        return doc.get("documentId") == DOC_ID
    except Exception as e:
        print(f"[TOKEN CHECK] gws CLI unavailable: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Document analysis helpers
# ---------------------------------------------------------------------------

def find_schedule_table(doc):
    """Return the schedule table element (5-col, largest row count)."""
    body = doc.get("body", {})
    best = None
    for elem in body.get("content", []):
        if "table" in elem:
            t = elem["table"]
            if t["columns"] == SCHEDULE_TABLE_COLS:
                if best is None or t["rows"] > best["table"]["rows"]:
                    best = elem
    if best is None:
        raise RuntimeError("Schedule table (5 columns) not found in document")
    return best


def get_row_text_ranges(table_elem):
    """
    Return a list of dicts, one per row, with:
      - row_index
      - cells: list of {text, start, end} per cell
      - row_start, row_end
      - has_strikethrough (bool)
    """
    rows = []
    for ri, row in enumerate(table_elem["table"]["tableRows"]):
        cells = []
        has_strike = False
        for cell in row.get("tableCells", []):
            cell_text = ""
            cell_start = None
            cell_end = None
            for cp in cell.get("content", []):
                if "paragraph" in cp:
                    for e in cp["paragraph"].get("elements", []):
                        if "textRun" in e:
                            cell_text += e["textRun"]["content"]
                            if cell_start is None:
                                cell_start = e["startIndex"]
                            cell_end = e["endIndex"]
                            if e["textRun"].get("textStyle", {}).get("strikethrough"):
                                has_strike = True
            cells.append({
                "text": cell_text.strip(),
                "start": cell_start,
                "end": cell_end,
            })
        rows.append({
            "row_index": ri,
            "cells": cells,
            "row_start": row.get("startIndex"),
            "row_end": row.get("endIndex"),
            "has_strikethrough": has_strike,
        })
    return rows


def find_row_by_text(rows, match_text):
    """Find a row whose concatenated cell text contains match_text (case-insensitive)."""
    match_lower = match_text.lower()
    for r in rows:
        full = " | ".join(c["text"] for c in r["cells"]).lower()
        if match_lower in full:
            return r
    return None


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def handle_add_row(item):
    """
    Add a new row to the bottom of the schedule table.
    item['row_data'] is a JSON string with keys: dates, item, location, status, notes
    """
    data = json.loads(item["row_data"])
    cell_values = [
        data.get("dates", ""),
        data.get("item", ""),
        data.get("location", ""),
        data.get("status", ""),
        data.get("notes", "Auto-added by doc sync processor"),
    ]

    # Fetch fresh doc to get current indices
    doc = gws_get_doc()
    table_elem = find_schedule_table(doc)
    table = table_elem["table"]
    last_row_index = table["rows"] - 1

    # Step 1: Insert an empty row below the last row
    insert_req = {
        "insertTableRow": {
            "tableCellLocation": {
                "tableStartLocation": {"index": table_elem["startIndex"]},
                "rowIndex": last_row_index,
                "columnIndex": 0,
            },
            "insertBelow": True,
        }
    }
    gws_batch_update([insert_req])

    # Step 2: Re-fetch doc to get updated indices after row insertion
    doc = gws_get_doc()
    table_elem = find_schedule_table(doc)
    table = table_elem["table"]
    new_row_index = table["rows"] - 1
    new_row = table["tableRows"][new_row_index]

    # Step 3: Insert text into each cell of the new row
    # Process cells in reverse order to avoid index shifting
    text_requests = []
    for ci in range(SCHEDULE_TABLE_COLS - 1, -1, -1):
        cell = new_row["tableCells"][ci]
        # Find the paragraph start index inside the cell
        for cp in cell.get("content", []):
            if "paragraph" in cp:
                for e in cp["paragraph"].get("elements", []):
                    insert_index = e["startIndex"]
                    break
                break
        text_requests.append({
            "insertText": {
                "location": {"index": insert_index},
                "text": cell_values[ci],
            }
        })

    if text_requests:
        gws_batch_update(text_requests)

    return f"Added row: {cell_values}"


def handle_strikethrough(item):
    """Apply strikethrough to a row matching item['match_text']."""
    doc = gws_get_doc()
    table_elem = find_schedule_table(doc)
    rows = get_row_text_ranges(table_elem)

    target = find_row_by_text(rows, item["match_text"])
    if target is None:
        raise RuntimeError(f"No row found matching: {item['match_text']}")

    if target["has_strikethrough"]:
        return f"Row already has strikethrough: {item['match_text']}"

    # Apply strikethrough to all text in the row
    requests = []
    for cell in target["cells"]:
        if cell["start"] is not None and cell["end"] is not None and cell["text"]:
            requests.append({
                "updateTextStyle": {
                    "range": {
                        "startIndex": cell["start"],
                        "endIndex": cell["end"],
                    },
                    "textStyle": {"strikethrough": True},
                    "fields": "strikethrough",
                }
            })

    if requests:
        gws_batch_update(requests)
        return f"Applied strikethrough to row: {item['match_text']}"
    else:
        return f"No text content found in row: {item['match_text']}"


def handle_remove_strikethrough(item):
    """Remove strikethrough from a row matching item['match_text']."""
    doc = gws_get_doc()
    table_elem = find_schedule_table(doc)
    rows = get_row_text_ranges(table_elem)

    target = find_row_by_text(rows, item["match_text"])
    if target is None:
        raise RuntimeError(f"No row found matching: {item['match_text']}")

    if not target["has_strikethrough"]:
        return f"Row does not have strikethrough: {item['match_text']}"

    # Remove strikethrough from all text in the row
    requests = []
    for cell in target["cells"]:
        if cell["start"] is not None and cell["end"] is not None and cell["text"]:
            requests.append({
                "updateTextStyle": {
                    "range": {
                        "startIndex": cell["start"],
                        "endIndex": cell["end"],
                    },
                    "textStyle": {"strikethrough": False},
                    "fields": "strikethrough",
                }
            })

    if requests:
        gws_batch_update(requests)
        return f"Removed strikethrough from row: {item['match_text']}"
    else:
        return f"No text content found in row: {item['match_text']}"


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

ACTION_HANDLERS = {
    "add_row": handle_add_row,
    "strikethrough": handle_strikethrough,
    "remove_strikethrough": handle_remove_strikethrough,
}


def process_queue():
    """Main entry point: process all pending items in the doc_sync_queue."""
    print("=" * 60)
    print(f"Life Planner Doc Sync Processor")
    print(f"Started at: {datetime.utcnow().isoformat()}Z")
    print(f"Document ID: {DOC_ID}")
    print(f"Database: {DB_PATH}")
    print("=" * 60)

    # Check gws CLI availability first
    if not check_gws_token():
        print("\n[FATAL] gws CLI token is expired or unavailable.")
        print("Items remain in the queue for the next run.")
        sys.exit(1)
    print("\n[OK] gws CLI token is valid.\n")

    # Ensure database and table exist
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Create table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_sync_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL CHECK(action IN ('add_row', 'strikethrough', 'remove_strikethrough')),
            table_name TEXT NOT NULL DEFAULT 'schedule',
            row_data TEXT,
            match_text TEXT,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'done', 'failed')),
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP
        )
    """)
    conn.commit()

    # Fetch pending items (including failed items that haven't exceeded max retries)
    cur.execute("""
        SELECT * FROM doc_sync_queue
        WHERE status IN ('pending', 'failed')
          AND retry_count < max_retries
        ORDER BY created_at ASC
    """)
    items = cur.fetchall()

    if not items:
        print("[INFO] No pending items in the doc_sync_queue. Nothing to process.")
        conn.close()
        return

    print(f"[INFO] Found {len(items)} pending item(s) to process.\n")

    stats = {"success": 0, "failed": 0, "skipped": 0}

    for item in items:
        item_dict = dict(item)
        item_id = item_dict["id"]
        action = item_dict["action"]
        print(f"--- Processing item #{item_id}: action={action} "
              f"(retry {item_dict['retry_count']}/{item_dict['max_retries']}) ---")

        handler = ACTION_HANDLERS.get(action)
        if handler is None:
            print(f"  [SKIP] Unknown action: {action}")
            stats["skipped"] += 1
            continue

        # Mark as processing
        cur.execute(
            "UPDATE doc_sync_queue SET status = 'processing' WHERE id = ?",
            (item_id,)
        )
        conn.commit()

        try:
            result_msg = handler(item_dict)
            # Mark as done
            cur.execute(
                "UPDATE doc_sync_queue SET status = 'done', processed_at = ?, error_message = NULL WHERE id = ?",
                (datetime.utcnow().isoformat(), item_id)
            )
            conn.commit()
            print(f"  [SUCCESS] {result_msg}")
            stats["success"] += 1

        except Exception as e:
            error_msg = str(e)
            new_retry = item_dict["retry_count"] + 1
            new_status = "failed" if new_retry < item_dict["max_retries"] else "failed"
            cur.execute(
                "UPDATE doc_sync_queue SET status = ?, retry_count = ?, error_message = ? WHERE id = ?",
                (new_status, new_retry, error_msg, item_id)
            )
            conn.commit()
            print(f"  [FAILED] {error_msg}")
            if new_retry < item_dict["max_retries"]:
                print(f"  [RETRY] Will retry ({new_retry}/{item_dict['max_retries']})")
                # Retry immediately
                try:
                    print(f"  [RETRY] Attempting retry now...")
                    time.sleep(1)  # Brief pause before retry
                    result_msg = handler(item_dict)
                    cur.execute(
                        "UPDATE doc_sync_queue SET status = 'done', retry_count = ?, processed_at = ?, error_message = NULL WHERE id = ?",
                        (new_retry, datetime.utcnow().isoformat(), item_id)
                    )
                    conn.commit()
                    print(f"  [SUCCESS on retry] {result_msg}")
                    stats["success"] += 1
                    stats["failed"] -= 0  # Don't double-count
                    continue
                except Exception as e2:
                    retry2 = new_retry + 1
                    cur.execute(
                        "UPDATE doc_sync_queue SET retry_count = ?, error_message = ? WHERE id = ?",
                        (retry2, str(e2), item_id)
                    )
                    conn.commit()
                    print(f"  [RETRY FAILED] {e2}")
                    if retry2 < item_dict["max_retries"]:
                        # One more retry
                        try:
                            print(f"  [RETRY] Final attempt ({retry2}/{item_dict['max_retries']})...")
                            time.sleep(2)
                            result_msg = handler(item_dict)
                            cur.execute(
                                "UPDATE doc_sync_queue SET status = 'done', retry_count = ?, processed_at = ?, error_message = NULL WHERE id = ?",
                                (retry2, datetime.utcnow().isoformat(), item_id)
                            )
                            conn.commit()
                            print(f"  [SUCCESS on final retry] {result_msg}")
                            stats["success"] += 1
                            continue
                        except Exception as e3:
                            cur.execute(
                                "UPDATE doc_sync_queue SET status = 'failed', retry_count = ?, error_message = ? WHERE id = ?",
                                (retry2 + 1, str(e3), item_id)
                            )
                            conn.commit()
                            print(f"  [FINAL FAILURE] {e3}")
            stats["failed"] += 1

    print("\n" + "=" * 60)
    print(f"Processing complete at {datetime.utcnow().isoformat()}Z")
    print(f"  Success: {stats['success']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")
    print("=" * 60)

    # Show final queue status
    cur.execute("SELECT status, COUNT(*) FROM doc_sync_queue GROUP BY status")
    for status, count in cur.fetchall():
        print(f"  Queue [{status}]: {count}")

    conn.close()


if __name__ == "__main__":
    process_queue()
