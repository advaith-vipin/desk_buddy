"""
Separate SQLite databases for the Desk Buddy.

Two database files:
  * facts.db        — user facts/preferences (likes, habits, info about the user)
  * tasks_reminders.db — tasks AND reminders in one table (kind='task' or 'reminder')
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone

from .._paths import FACTS_DB_PATH, TASKS_DB_PATH, TMP_DIR, ROOT

_facts_connection = None
_tasks_connection = None
_facts_lock = threading.Lock()
_tasks_lock = threading.Lock()


def _get_facts_connection():
    """Lazily create and return facts database connection."""
    global _facts_connection
    with _facts_lock:
        if _facts_connection is None:
            _facts_connection = sqlite3.connect(FACTS_DB_PATH, check_same_thread=False, timeout=10.0)
            _facts_connection.row_factory = sqlite3.Row
            try:
                _facts_connection.execute("PRAGMA journal_mode=WAL;")
                _facts_connection.execute("PRAGMA synchronous=NORMAL;")
                _facts_connection.execute("PRAGMA busy_timeout=5000;")
            except Exception:
                pass
            _facts_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL UNIQUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Index for faster lookups
            _facts_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_text ON facts(lower(text))"
            )
            _facts_connection.commit()
        return _facts_connection


def _get_tasks_connection():
    """Lazily create and return tasks/reminders database connection."""
    global _tasks_connection
    with _tasks_lock:
        if _tasks_connection is None:
            _tasks_connection = sqlite3.connect(TASKS_DB_PATH, check_same_thread=False, timeout=10.0)
            _tasks_connection.row_factory = sqlite3.Row
            try:
                _tasks_connection.execute("PRAGMA journal_mode=WAL;")
                _tasks_connection.execute("PRAGMA synchronous=NORMAL;")
                _tasks_connection.execute("PRAGMA busy_timeout=5000;")
            except Exception:
                pass
            _tasks_connection.execute(
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
            _tasks_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_kind ON items(kind)"
            )
            _tasks_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_when ON items(when_at)"
            )
            _tasks_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_completed ON items(completed)"
            )
            _tasks_connection.commit()
        return _tasks_connection


def init_databases():
    """Initialise both databases."""
    _ = _get_facts_connection()
    _ = _get_tasks_connection()


def close_databases():
    """Close both database connections."""
    global _facts_connection, _tasks_connection
    with _facts_lock:
        if _facts_connection is not None:
            _facts_connection.close()
            _facts_connection = None
    with _tasks_lock:
        if _tasks_connection is not None:
            _tasks_connection.close()
            _tasks_connection = None


# ---------------------------------------------------------------------------
# Facts API
# ---------------------------------------------------------------------------

def save_fact(text):
    """Save a fact. Returns (id, created) — created=False if duplicate."""
    text = (text or "").strip()
    if not text:
        return None, False
    conn = _get_facts_connection()
    norm = " ".join(text.lower().split())
    with _facts_lock:
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
    conn = _get_facts_connection()
    with _facts_lock:
        cur = conn.execute("SELECT id, text FROM facts ORDER BY id")
        return [dict(r) for r in cur.fetchall()]


def delete_fact(fact_id):
    """Delete a fact by ID. Returns True if it existed."""
    conn = _get_facts_connection()
    with _facts_lock:
        cur = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        conn.commit()
    return cur.rowcount > 0


def get_fact(fact_id):
    """Return a single fact dict by ID, or None if not found."""
    conn = _get_facts_connection()
    with _facts_lock:
        cur = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def update_fact(fact_id, text):
    """Update a fact's text. Returns updated dict or None if not found."""
    text = (text or "").strip()
    if not text:
        return None
    conn = _get_facts_connection()
    norm = " ".join(text.lower().split())
    with _facts_lock:
        # Check for duplicate (excluding self)
        cur = conn.execute("SELECT id FROM facts WHERE lower(text) = ? AND id != ?", (norm, fact_id))
        if cur.fetchone():
            return None  # duplicate
        cur = conn.execute("UPDATE facts SET text = ? WHERE id = ?", (text, fact_id))
        conn.commit()
        if cur.rowcount == 0:
            return None
        cur = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def clear_facts():
    """Delete all facts."""
    conn = _get_facts_connection()
    with _facts_lock:
        conn.execute("DELETE FROM facts")
        conn.commit()


# ---------------------------------------------------------------------------
# Tasks + Reminders API (unified items table)
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
    conn = _get_tasks_connection()
    now = datetime.now(timezone.utc).isoformat()
    when_sql = _normalize_when(when_at)

    with _tasks_lock:
        cur = conn.execute(
            """
            INSERT INTO items (kind, title, description, priority, when_at, completed, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (kind, title, description, priority, when_sql, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM items WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row) if row else None


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
    conn = _get_tasks_connection()
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

    with _tasks_lock:
        cur = conn.execute(f"SELECT * FROM items {where} {order}", params)
        return [dict(r) for r in cur.fetchall()]


def get_tasks(completed=None, priority=None):
    """Return only tasks (back-compat)."""
    return get_items(kind="task", completed=completed, priority=priority)


def get_reminders(completed=None, priority=None):
    """Return only reminders (back-compat)."""
    return get_items(kind="reminder", completed=completed, priority=priority)


def get_item(item_id):
    """Return a single item dict by ID, or None if not found."""
    conn = _get_tasks_connection()
    with _tasks_lock:
        cur = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,))
        row = cur.fetchone()
    return dict(row) if row else None


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
    conn = _get_tasks_connection()

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

    with _tasks_lock:
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
    conn = _get_tasks_connection()
    with _tasks_lock:
        cur = conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        conn.commit()
    return cur.rowcount > 0


def delete_task(task_id):
    """Back-compat."""
    return delete_item(task_id)


def delete_reminder(reminder_id):
    """Back-compat."""
    return delete_item(reminder_id)


def clear_tasks_reminders():
    """Delete all tasks and reminders."""
    conn = _get_tasks_connection()
    with _tasks_lock:
        conn.execute("DELETE FROM items")
        conn.commit()


# ---------------------------------------------------------------------------
# Migration helpers (one-time)
# ---------------------------------------------------------------------------

def migrate_from_old_schema():
    """
    Move legacy data from old tables (tasks, reminders) and facts folder
    into the new unified schema. Idempotent — safe to call repeatedly.
    """
    moved = {"items": 0, "facts": 0}

    # Migrate old 'tasks' table from old DB
    old_db_path = os.path.join(ROOT, "task_reminder.db")
    if os.path.exists(old_db_path):
        old_conn = sqlite3.connect(old_db_path, check_same_thread=False)
        old_conn.row_factory = sqlite3.Row
        tasks_conn = _get_tasks_connection()
        facts_conn = _get_facts_connection()

        with tasks_conn:
            # Migrate old 'tasks' table
            try:
                cur = old_conn.execute("SELECT * FROM tasks")
                for row in cur.fetchall():
                    exists = tasks_conn.execute(
                        "SELECT 1 FROM items WHERE kind='task' AND title=? AND when_at=?",
                        (row["title"], row["due_date"])).fetchone()
                    if not exists:
                        tasks_conn.execute(
                            "INSERT INTO items (kind, title, description, priority, when_at, completed, created_at) "
                            "VALUES ('task', ?, ?, ?, ?, ?, ?)",
                            (row["title"], row["description"], row["priority"],
                             row["due_date"], row["completed"], row["created_at"]))
                        moved["items"] += 1
                tasks_conn.execute("DROP TABLE IF EXISTS tasks")
            except sqlite3.OperationalError:
                pass

            # Migrate old 'reminders' table
            try:
                cur = old_conn.execute("SELECT * FROM reminders")
                for row in cur.fetchall():
                    exists = tasks_conn.execute(
                        "SELECT 1 FROM items WHERE kind='reminder' AND title=? AND when_at=?",
                        (row["title"], row["scheduled_at"])).fetchone()
                    if not exists:
                        tasks_conn.execute(
                            "INSERT INTO items (kind, title, description, priority, when_at, completed, created_at) "
                            "VALUES ('reminder', ?, ?, ?, ?, ?, ?)",
                            (row["title"], row["description"], row["priority"],
                             row["scheduled_at"], row["completed"], row["created_at"]))
                        moved["items"] += 1
                tasks_conn.execute("DROP TABLE IF EXISTS reminders")
            except sqlite3.OperationalError:
                pass

            # Migrate old 'facts' table
            try:
                cur = old_conn.execute("SELECT * FROM facts")
                for row in cur.fetchall():
                    save_fact(row["text"])
                    moved["facts"] += 1
                facts_conn.execute("DROP TABLE IF EXISTS facts")
            except sqlite3.OperationalError:
                pass

        old_conn.close()

    # Facts folder migration (if it exists)
    facts_dir = os.path.join(TMP_DIR, "facts")
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