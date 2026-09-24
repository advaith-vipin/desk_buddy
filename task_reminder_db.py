"""
Simple SQLite-based storage for the Desk Buddy.

Two tables:
  * items   — tasks AND reminders in one table (kind='task' or 'reminder')
  * facts   — user facts/preferences (likes, habits, info about the user)

Both use the same DB file: task_reminder.db
"""

import sqlite3
import os
import threading
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "task_reminder.db")

_connection = None
_lock = threading.Lock()


def _get_connection():
    """Lazily create and return a single connection (thread-safe)."""
    global _connection
    with _lock:
        if _connection is None:
            _connection = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
            _connection.row_factory = sqlite3.Row
            # Faster + more robust on Pi: WAL mode avoids readonly locks during delete
            try:
                _connection.execute("PRAGMA journal_mode=WAL;")
                _connection.execute("PRAGMA synchronous=NORMAL;")
                _connection.execute("PRAGMA busy_timeout=5000;")
            except Exception:
                pass

            # Unified items table: tasks + reminders
            _connection.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL CHECK(kind IN ('task','reminder')),
                    title TEXT NOT NULL,
                    description TEXT,
                    completed INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    when_at DATETIME,          -- due_date for tasks, scheduled_at for reminders
                    priority INTEGER DEFAULT 1
                )
                """
            )
            # Facts table: user preferences / info about the user
            _connection.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Indexes for speed
            try:
                _connection.execute("CREATE INDEX IF NOT EXISTS idx_items_kind ON items(kind)")
                _connection.execute("CREATE INDEX IF NOT EXISTS idx_items_completed ON items(completed)")
                _connection.execute("CREATE INDEX IF NOT EXISTS idx_items_when ON items(when_at)")
                _connection.execute("CREATE INDEX IF NOT EXISTS idx_facts_text ON facts(lower(text))")
            except Exception:
                pass
            _connection.commit()
        return _connection


def init_db():
    """Initialise the database (create tables if they do not exist)."""
    _ = _get_connection()


def close_db():
    """Close the database connection."""
    global _connection
    with _lock:
        if _connection is not None:
            try:
                _connection.close()
            except Exception:
                pass
            _connection = None

def _row_to_dict(row):
    """Convert sqlite Row to dict with backward-compat aliases."""
    if row is None:
        return None
    d = dict(row)
    # Aliases for old tests / callers expecting due_date / scheduled_at / dueDate
    when = d.get("when_at")
    d["due_date"] = when
    d["scheduled_at"] = when
    d["dueDate"] = when
    # Ensure description always string
    if d.get("description") is None:
        d["description"] = ""
    return d


# ---------------------------------------------------------------------------
# Items (tasks + reminders) — unified API
# ---------------------------------------------------------------------------

def _normalize_when(val):
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return val


def create_item(kind, title, description="", priority=1, when_at=None):
    """
    Create a task or reminder.

    Args:
        kind: 'task' or 'reminder'
        title: str — required
        description: str — optional
        priority: int — 1=low, 2=medium, 3=high (default: 1)
        when_at: str or datetime or None — due date (task) or scheduled time (reminder)

    Returns:
        dict with keys: id, kind, title, description, completed, created_at, when_at, priority
    """
    assert kind in ("task", "reminder")
    conn = _get_connection()
    now = datetime.now(timezone.utc).isoformat()
    when_sql = _normalize_when(when_at)

    with _lock:
        cur = conn.execute(
            """
            INSERT INTO items (kind, title, description, priority, when_at, completed, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (kind, title, description, priority, when_sql, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM items WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _row_to_dict(row)


def create_task(title, description="", priority=1, due_date=None):
    """Create a task (convenience wrapper)."""
    return create_item("task", title, description, priority, due_date)


def create_reminder(title, description="", priority=1, scheduled_at=None):
    """Create a reminder (convenience wrapper)."""
    return create_item("reminder", title, description, priority, scheduled_at)


def get_items(kind=None, completed=None, priority=None):
    """
    Return a list of item dicts, optionally filtered.

    Args:
        kind: 'task', 'reminder', or None for both
        completed: bool or None — filter by completed status
        priority: int or None — filter by priority level

    Sorted: reminders by when_at ASC (soonest first), tasks by priority then created_at.
    """
    conn = _get_connection()
    conditions = []
    params = []

    if kind is not None:
        conditions.append("kind = ?")
        params.append(kind)

    if completed is not None:
        conditions.append("completed = ?")
        params.append(1 if completed else 0)

    if priority is not None:
        conditions.append("priority = ?")
        params.append(priority)

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    # Reminders: soonest first (NULLs last). Tasks: priority then newest first.
    order = "ORDER BY " + (
        "CASE WHEN kind='reminder' THEN 0 ELSE 1 END, "
        "CASE WHEN kind='reminder' THEN (when_at IS NULL) ELSE 2 END, "
        "when_at ASC, priority ASC, created_at DESC"
    )

    with _lock:
        cur = conn.execute(f"SELECT * FROM items {where} {order}", params)
        return [_row_to_dict(r) for r in cur.fetchall()]


def get_tasks(completed=None, priority=None):
    """Return only tasks (back-compat)."""
    return get_items(kind="task", completed=completed, priority=priority)


def get_reminders(completed=None, priority=None):
    """Return only reminders (back-compat)."""
    return get_items(kind="reminder", completed=completed, priority=priority)


def get_item(item_id):
    """Return a single item dict by ID, or None if not found."""
    conn = _get_connection()
    with _lock:
        cur = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,))
        row = cur.fetchone()
    return _row_to_dict(row)


def get_task(task_id):
    """Back-compat: fetch task by ID."""
    row = get_item(task_id)
    return row if row and row["kind"] == "task" else None


def get_reminder(reminder_id):
    """Back-compat: fetch reminder by ID."""
    row = get_item(reminder_id)
    return row if row and row["kind"] == "reminder" else None


def update_item(item_id, **updates):
    """Update item fields and return the updated dict (or None if not found)."""
    conn = _get_connection()

    # Map legacy aliases to canonical when_at
    if "due_date" in updates and "when_at" not in updates:
        updates["when_at"] = updates.pop("due_date")
    if "scheduled_at" in updates and "when_at" not in updates:
        updates["when_at"] = updates.pop("scheduled_at")
    if "dueDate" in updates and "when_at" not in updates:
        updates["when_at"] = updates.pop("dueDate")

    allowed = ("title", "description", "priority", "when_at", "completed", "kind")
    set_parts = []
    params = []

    for key in allowed:
        if key in updates:
            if key == "completed":
                set_parts.append("completed = ?")
                params.append(1 if updates[key] else 0)
            elif key == "when_at":
                set_parts.append("when_at = ?")
                params.append(_normalize_when(updates[key]))
            elif key == "kind":
                if updates[key] not in ("task", "reminder"):
                    raise ValueError("kind must be 'task' or 'reminder'")
                set_parts.append("kind = ?")
                params.append(updates[key])
            else:
                set_parts.append(f"{key} = ?")
                params.append(updates[key])

    if not set_parts:
        return get_item(item_id)

    params.append(item_id)
    set_clause = ", ".join(set_parts)

    with _lock:
        conn.execute(f"UPDATE items SET {set_clause} WHERE id = ?", params)
        conn.commit()

    return get_item(item_id)


def update_task(task_id, **updates):
    """Back-compat: update a task."""
    return update_item(task_id, **updates)


def update_reminder(reminder_id, **updates):
    """Back-compat: update a reminder."""
    return update_item(reminder_id, **updates)


def delete_item(item_id):
    """Delete an item by ID. Returns True if deleted."""
    conn = _get_connection()
    with _lock:
        cur = conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        conn.commit()
    return cur.rowcount > 0


def delete_task(task_id):
    """Back-compat."""
    return delete_item(task_id)


def delete_reminder(reminder_id):
    """Back-compat."""
    return delete_item(reminder_id)


# ---------------------------------------------------------------------------
# Facts — user info/preferences
# ---------------------------------------------------------------------------

def save_fact(text):
    """Save a fact. Returns (id, created) — created=False if duplicate."""
    text = (text or "").strip()
    if not text:
        return None, False
    conn = _get_connection()
    norm = " ".join(text.lower().split())
    with _lock:
        cur = conn.execute("SELECT id FROM facts WHERE lower(text) = ?", (norm,))
        row = cur.fetchone()
        if row:
            return row["id"], False
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO facts (text, created_at) VALUES (?, ?)", (text, now))
        conn.commit()
        return cur.lastrowid, True


def list_facts():
    """Return all facts as [{'id': int, 'text': str}, ...] ordered by id."""
    conn = _get_connection()
    with _lock:
        cur = conn.execute("SELECT id, text FROM facts ORDER BY id")
        return [dict(r) for r in cur.fetchall()]


def delete_fact(fact_id):
    """Delete a fact by ID. Returns True if it existed."""
    conn = _get_connection()
    with _lock:
        cur = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        conn.commit()
    return cur.rowcount > 0


def clear_all():
    """Delete every item and fact (used by --reset-memory)."""
    conn = _get_connection()
    with _lock:
        conn.execute("DELETE FROM items")
        conn.execute("DELETE FROM facts")
        conn.commit()


# ---------------------------------------------------------------------------
# Migration helpers (one-time)
# ---------------------------------------------------------------------------

def migrate_from_old_schema():
    """
    Move legacy data from old tables (tasks, reminders) and facts folder
    into the new unified schema. Idempotent — safe to call repeatedly.
    """
    conn = _get_connection()
    moved = {"items": 0, "facts": 0}

    with _lock:
        # Migrate old 'tasks' table
        try:
            cur = conn.execute("SELECT * FROM tasks")
            for row in cur.fetchall():
                # Skip if already migrated (check by title+when_at+kind)
                exists = conn.execute(
                    "SELECT 1 FROM items WHERE kind='task' AND title=? AND when_at=?",
                    (row["title"], row["due_date"])).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO items (kind, title, description, priority, when_at, completed, created_at) "
                        "VALUES ('task', ?, ?, ?, ?, ?, ?)",
                        (row["title"], row["description"], row["priority"],
                         row["due_date"], row["completed"], row["created_at"]))
                    moved["items"] += 1
            # Drop old table
            conn.execute("DROP TABLE IF EXISTS tasks")
        except sqlite3.OperationalError:
            pass  # table doesn't exist

        # Migrate old 'reminders' table
        try:
            cur = conn.execute("SELECT * FROM reminders")
            for row in cur.fetchall():
                exists = conn.execute(
                    "SELECT 1 FROM items WHERE kind='reminder' AND title=? AND when_at=?",
                    (row["title"], row["scheduled_at"])).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO items (kind, title, description, priority, when_at, completed, created_at) "
                        "VALUES ('reminder', ?, ?, ?, ?, ?, ?)",
                        (row["title"], row["description"], row["priority"],
                         row["scheduled_at"], row["completed"], row["created_at"]))
                    moved["items"] += 1
            conn.execute("DROP TABLE IF EXISTS reminders")
        except sqlite3.OperationalError:
            pass

        conn.commit()

    # Facts folder migration (if it exists)
    facts_dir = os.path.join(BASE_DIR, ".tmp", "facts")
    if os.path.isdir(facts_dir):
        for fn in sorted(os.listdir(facts_dir)):
            if fn.endswith(".txt"):
                path = os.path.join(facts_dir, fn)
                try:
                    with open(path, encoding="utf-8") as f:
                        text = f.read().strip()
                    if text:
                        save_fact(text)
                        moved["facts"] += 1
                except OSError:
                    pass

    return moved