"""
Test suite for task_reminder_db.py
exercises all CRUD methods.
Run with: python -m pytest test_task_reminder.py -v
or: python test_task_reminder.py
"""

import os
import sys

# Ensure the project package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_reminder_db import (
    init_db,
    create_task,
    get_tasks,
    get_task,
    update_task,
    delete_task,
    close_db)


# Use the SAME DB file the module uses, so reset/teardown works correctly
MODULE_DIR = os.path.dirname(os.path.abspath("task_reminder_db.py"))
DB_PATH = os.path.join(MODULE_DIR, "task_reminder.db")


def reset_db():
    """Remove the module's DB file so tests start fresh each time."""
    # Ensure connection is closed first to avoid readonly/WAL lock on Pi
    try:
        close_db()
    except Exception:
        pass
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = DB_PATH + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def setup_module():
    """Called once before any tests in this module."""
    reset_db()
    init_db()


def teardown_module():
    """Called once after all tests in this module."""
    close_db()
    reset_db()


class TestTaskReminderCRUD:
    """Smoke‑test all CRUD operations."""

    def setup_method(self):
        reset_db()
        init_db()

    def teardown_method(self):
        close_db()
        reset_db()

    def test_create_and_read(self):
        """Create a task and immediately read it back."""
        t = create_task(title="Buy milk", description="Get 2% milk", priority=2)
        assert t is not None
        assert t["title"] == "Buy milk"
        assert t["description"] == "Get 2% milk"
        assert t["completed"] == 0
        assert t["priority"] == 2
        assert t["id"] > 0

        # Read it back via get_task
        t2 = get_task(t["id"])
        assert t2 is not None
        assert t2["title"] == "Buy milk"

    def test_get_tasks_all(self):
        """get_tasks() with no filters returns everything."""
        create_task(title="Task A", priority=1)
        create_task(title="Task B", priority=2)
        create_task(title="Task C", priority=3)

        all_tasks = get_tasks()
        assert len(all_tasks) == 3
        titles = [t["title"] for t in all_tasks]
        assert "Task A" in titles
        assert "Task B" in titles
        assert "Task C" in titles

    def test_get_tasks_filter_completed(self):
        """get_tasks(completed=True) only returns completed tasks."""
        create_task(title="Active", priority=1)
        t2 = create_task(title="Done", priority=2)
        update_task(t2["id"], completed=True)

        active = get_tasks(completed=False)
        assert len(active) == 1
        assert active[0]["title"] == "Active"

        done = get_tasks(completed=True)
        assert len(done) == 1
        assert done[0]["title"] == "Done"

    def test_get_tasks_filter_priority(self):
        """get_tasks(priority=2) filters by priority."""
        create_task(title="Low", priority=1)
        create_task(title="Medium", priority=2)
        create_task(title="High", priority=3)

        medium = get_tasks(priority=2)
        assert len(medium) == 1
        assert medium[0]["title"] == "Medium"

    def test_update_task_fields(self):
        """update_task can modify any field."""
        t = create_task(title="Original", priority=1)
        assert t["title"] == "Original"
        assert t["priority"] == 1

        # Update title and priority
        updated = update_task(t["id"], title="Renamed", priority=3)
        assert updated["title"] == "Renamed"
        assert updated["priority"] == 3

        # Verify via get_task
        fetched = get_task(t["id"])
        assert fetched["title"] == "Renamed"
        assert fetched["priority"] == 3

    def test_update_task_completed(self):
        """update_task can toggle completed flag."""
        t = create_task(title="Toggle me", priority=1)
        assert t["completed"] == 0

        update_task(t["id"], completed=True)
        fetched = get_task(t["id"])
        assert fetched["completed"] == 1

        update_task(t["id"], completed=False)
        fetched = get_task(t["id"])
        assert fetched["completed"] == 0

    def test_delete_task(self):
        """delete_task removes the task and returns True."""
        t = create_task(title="To be deleted", priority=1)
        assert get_task(t["id"]) is not None

        result = delete_task(t["id"])
        assert result is True
        assert get_task(t["id"]) is None

    def test_delete_task_nonexistent(self):
        """delete_task on a non‑existent ID returns False."""
        result = delete_task(9999)
        assert result is False

    def test_due_date(self):
        """create_task accepts a due_date datetime."""
        from datetime import datetime, timezone
        dt = datetime(2026, 12, 25, tzinfo=timezone.utc)
        t = create_task(title="Christmas", due_date=dt, priority=1)
        # Backward-compat alias: both due_date and when_at should be set
        assert t.get("due_date") is not None or t.get("when_at") is not None
        # fetched via get_task should also have it
        fetched = get_task(t["id"])
        assert fetched.get("due_date") is not None or fetched.get("when_at") is not None

    def test_persistence_across_operations(self):
        """Full round‑trip: create → read → update → delete → verify gone."""
        # init_db already called in setup_method

        # Create
        t1 = create_task(title="First", priority=1)
        t2 = create_task(title="Second", priority=2)

        # Read all
        all_tasks = get_tasks()
        assert len(all_tasks) == 2

        # Update first
        update_task(t1["id"], title="Renamed", completed=True)
        assert get_task(t1["id"])["title"] == "Renamed"
        assert get_task(t1["id"])["completed"] == 1

        # Delete second
        delete_task(t2["id"])
        remaining = get_tasks()
        assert len(remaining) == 1
        assert remaining[0]["title"] == "Renamed"

    def test_empty_database(self):
        """get_tasks() on fresh DB returns empty list."""
        reset_db()
        init_db()
        tasks = get_tasks()
        assert len(tasks) == 0


if __name__ == "__main__":
    # Allow running without pytest: python test_task_reminder.py
    print("Running CRUD tests...")

    import __main__ as m
    if hasattr(m, 'TestTaskReminderCRUD'):
        # We're being run directly, run all tests
        setup_module()
        t = TestTaskReminderCRUD()
        t.setup_method()

        tests = [
            'test_create_and_read',
            'test_get_tasks_all',
            'test_get_tasks_filter_completed',
            'test_get_tasks_filter_priority',
            'test_update_task_fields',
            'test_update_task_completed',
            'test_delete_task',
            'test_delete_task_nonexistent',
            'test_due_date',
            'test_persistence_across_operations',
            'test_empty_database',
        ]

        passed = 0
        failed = 0
        for test_name in tests:
            try:
                getattr(t, test_name)()
                print(f'  PASS: {test_name}')
                passed += 1
            except Exception as e:
                print(f'  FAIL: {test_name} — {e}')
                failed += 1

            t.teardown_method()
            reset_db()
            t.setup_method()

        teardown_module()
        print(f'\n{passed} passed, {failed} failed out of {len(tests)}')
    else:
        print('Import mode: import test_task_reminder and run tests manually')