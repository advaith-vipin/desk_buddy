"""
crud_tools.py — Unified CRUD API for Desk Buddy.

All data lives in SQLite:
  facts.db            -> facts table (user preferences, likes, info)
  tasks_reminders.db  -> items table (kind='task' or 'reminder')

This module wraps db.py with a clean, documented API covering EVERY
Create / Read / Update / Delete operation for Tasks, Reminders, and Facts.
Both the user (direct Python / CLI) and the LLM (Ollama tool calling /
natural-language via memory_handler) use the SAME functions.

Usage as library:
    import crud_tools as crud
    crud.task_create("buy milk", priority=2)
    crud.task_list()
    crud.fact_create("my favorite color is green")

Usage as CLI:
    python crud_tools.py task create "buy milk" --priority 2
    python crud_tools.py task list
    python crud_tools.py reminder create "call mom" --when "tomorrow at 5pm"
    python crud_tools.py fact list
    python crud_tools.py clear all

All functions return plain dicts/lists with no random IDs exposed to the user
in spoken replies — IDs are only for API/programmatic use.
"""

from datetime import datetime, timezone
from db import (
    create_task as _create_task,
    create_reminder as _create_reminder,
    create_item as _create_item,
    get_items as _get_items,
    get_item as _get_item,
    get_tasks as _get_tasks,
    get_reminders as _get_reminders,
    get_task as _get_task,
    get_reminder as _get_reminder,
    update_item as _update_item,
    update_task as _update_task,
    update_reminder as _update_reminder,
    delete_item as _delete_item,
    delete_task as _delete_task,
    delete_reminder as _delete_reminder,
    save_fact as _save_fact,
    list_facts as _list_facts,
    get_fact as _get_fact,
    update_fact as _update_fact,
    delete_fact as _delete_fact,
    clear_facts as _clear_facts,
    clear_tasks_reminders as _clear_tasks,
)

# ---------------------------------------------------------------------------
# Task CRUD
# ---------------------------------------------------------------------------

def task_create(title: str, description: str = "", priority: int = 1, when_at=None, due_date=None):
    """Create a task. due_date alias → when_at."""
    when = due_date if due_date is not None else when_at
    return _create_task(title=title, description=description, priority=priority, due_date=when)

def task_list(completed=None, priority=None, search: str = None):
    """List tasks, optionally filtered by completed/priority and substring search."""
    rows = _get_tasks(completed=completed, priority=priority)
    if search:
        low = search.lower()
        rows = [r for r in rows if low in (r.get("title") or "").lower() or low in (r.get("description") or "").lower()]
    return rows

def task_get(task_id: int):
    """Get one task by ID."""
    return _get_task(task_id)

def task_update(task_id: int, title=None, description=None, priority=None, when_at=None, due_date=None, completed=None):
    """Update any task field. due_date alias → when_at."""
    updates = {}
    if title is not None: updates["title"] = title
    if description is not None: updates["description"] = description
    if priority is not None: updates["priority"] = priority
    if when_at is not None: updates["when_at"] = when_at
    if due_date is not None: updates["when_at"] = due_date
    if completed is not None: updates["completed"] = completed
    return _update_task(task_id, **updates)

def task_delete(task_id: int):
    """Delete a task by ID."""
    return _delete_task(task_id)

def task_search(query: str):
    """Search tasks by title/description substring."""
    return task_list(search=query)

# ---------------------------------------------------------------------------
# Reminder CRUD
# ---------------------------------------------------------------------------

def reminder_create(title: str, description: str = "", priority: int = 1, when_at=None, scheduled_at=None):
    """Create a reminder. scheduled_at alias → when_at."""
    when = scheduled_at if scheduled_at is not None else when_at
    return _create_reminder(title=title, description=description, priority=priority, scheduled_at=when)

def reminder_list(completed=None, priority=None, search: str = None):
    """List reminders, optionally filtered."""
    rows = _get_reminders(completed=completed, priority=priority)
    if search:
        low = search.lower()
        rows = [r for r in rows if low in (r.get("title") or "").lower()]
    return rows

def reminder_get(reminder_id: int):
    """Get one reminder by ID."""
    return _get_reminder(reminder_id)

def reminder_update(reminder_id: int, title=None, description=None, priority=None, when_at=None, scheduled_at=None, completed=None):
    """Update any reminder field."""
    updates = {}
    if title is not None: updates["title"] = title
    if description is not None: updates["description"] = description
    if priority is not None: updates["priority"] = priority
    if when_at is not None: updates["when_at"] = when_at
    if scheduled_at is not None: updates["when_at"] = scheduled_at
    if completed is not None: updates["completed"] = completed
    return _update_reminder(reminder_id, **updates)

def reminder_delete(reminder_id: int):
    """Delete a reminder by ID."""
    return _delete_reminder(reminder_id)

def reminder_search(query: str):
    """Search reminders by title substring."""
    return reminder_list(search=query)

# ---------------------------------------------------------------------------
# Fact CRUD
# ---------------------------------------------------------------------------

def fact_create(text: str):
    """Create a fact (user preference/info). Returns (id, created)."""
    return _save_fact(text)

def fact_list(search: str = None):
    """List all facts, optionally filtered by substring."""
    rows = _list_facts()
    if search:
        low = search.lower()
        rows = [r for r in rows if low in (r.get("text") or "").lower()]
    return rows

def fact_get(fact_id: int):
    """Get one fact by ID."""
    return _get_fact(fact_id)

def fact_update(fact_id: int, text: str):
    """Update a fact's text."""
    return _update_fact(fact_id, text)

def fact_delete(fact_id: int):
    """Delete a fact by ID."""
    return _delete_fact(fact_id)

def fact_search(query: str):
    """Search facts by substring."""
    return fact_list(search=query)

# ---------------------------------------------------------------------------
# Unified Items (tasks + reminders together)
# ---------------------------------------------------------------------------

def item_create(kind: str, title: str, description: str = "", priority: int = 1, when_at=None):
    """Create either kind='task' or 'reminder'."""
    return _create_item(kind, title, description, priority, when_at)

def item_list(kind=None, completed=None, priority=None, search: str = None):
    """List all items (tasks+reminders), optionally filtered."""
    rows = _get_items(kind=kind, completed=completed, priority=priority)
    if search:
        low = search.lower()
        rows = [r for r in rows if low in (r.get("title") or "").lower()]
    return rows

def item_get(item_id: int):
    """Get one item (task or reminder) by ID."""
    return _get_item(item_id)

def item_update(item_id: int, **kwargs):
    """Update any item field."""
    return _update_item(item_id, **kwargs)

def item_delete(item_id: int):
    """Delete any item by ID."""
    return _delete_item(item_id)

def item_search(query: str):
    """Search all items by title substring."""
    return item_list(search=query)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clear_all():
    """Delete everything: tasks, reminders, and facts."""
    _clear_tasks()
    _clear_facts()
    return {"cleared": True}

def clear_tasks():
    """Delete all tasks + reminders."""
    _clear_tasks()
    return {"cleared": True}

def clear_facts_only():
    """Delete all facts."""
    _clear_facts()
    return {"cleared": True}

# ---------------------------------------------------------------------------
# Tool manifest for Ollama / LLM function calling
# ---------------------------------------------------------------------------

TOOLS = [
    {"name": "task_create", "description": "Create a new to-do task", "parameters": {"title": "str (required)", "description": "str optional", "priority": "int 1-3 optional", "when_at": "ISO datetime or natural language like 'tomorrow at 5pm' optional"}},
    {"name": "task_list", "description": "List tasks, optionally filtered by completed/priority/search", "parameters": {"completed": "bool optional", "priority": "int optional", "search": "str optional substring"}},
    {"name": "task_get", "description": "Get one task by ID", "parameters": {"task_id": "int required"}},
    {"name": "task_update", "description": "Update a task's fields", "parameters": {"task_id": "int required", "title": "str optional", "description": "str optional", "priority": "int optional", "completed": "bool optional", "when_at": "datetime optional"}},
    {"name": "task_delete", "description": "Delete a task by ID", "parameters": {"task_id": "int required"}},
    {"name": "task_search", "description": "Search tasks by substring", "parameters": {"query": "str required"}},
    {"name": "reminder_create", "description": "Create a time-based reminder", "parameters": {"title": "str required", "when_at": "datetime required (ISO or natural like 'tomorrow at 5pm')", "priority": "int optional"}},
    {"name": "reminder_list", "description": "List reminders", "parameters": {"completed": "bool optional", "search": "str optional"}},
    {"name": "reminder_get", "description": "Get one reminder by ID", "parameters": {"reminder_id": "int required"}},
    {"name": "reminder_update", "description": "Update a reminder", "parameters": {"reminder_id": "int required", "title": "str optional", "when_at": "datetime optional", "completed": "bool optional"}},
    {"name": "reminder_delete", "description": "Delete a reminder by ID", "parameters": {"reminder_id": "int required"}},
    {"name": "reminder_search", "description": "Search reminders by substring", "parameters": {"query": "str required"}},
    {"name": "fact_create", "description": "Save a fact about the user (likes, preferences, info)", "parameters": {"text": "str required"}},
    {"name": "fact_list", "description": "List all facts about the user", "parameters": {"search": "str optional"}},
    {"name": "fact_get", "description": "Get one fact by ID", "parameters": {"fact_id": "int required"}},
    {"name": "fact_update", "description": "Update a fact's text", "parameters": {"fact_id": "int required", "text": "str required"}},
    {"name": "fact_delete", "description": "Delete a fact by ID", "parameters": {"fact_id": "int required"}},
    {"name": "fact_search", "description": "Search facts by substring", "parameters": {"query": "str required"}},
    {"name": "item_list", "description": "List all items (tasks+reminders) with optional filters", "parameters": {"kind": "'task'|'reminder' optional", "completed": "bool optional", "search": "str optional"}},
    {"name": "item_search", "description": "Search all tasks+reminders by substring", "parameters": {"query": "str required"}},
    {"name": "clear_all", "description": "Delete ALL tasks, reminders, and facts", "parameters": {}},
]

TOOL_FUNCS = {
    "task_create": task_create, "task_list": task_list, "task_get": task_get,
    "task_update": task_update, "task_delete": task_delete, "task_search": task_search,
    "reminder_create": reminder_create, "reminder_list": reminder_list, "reminder_get": reminder_get,
    "reminder_update": reminder_update, "reminder_delete": reminder_delete, "reminder_search": reminder_search,
    "fact_create": fact_create, "fact_list": fact_list, "fact_get": fact_get,
    "fact_update": fact_update, "fact_delete": fact_delete, "fact_search": fact_search,
    "item_list": item_list, "item_search": item_search, "item_get": item_get,
    "item_update": item_update, "item_delete": item_delete,
    "clear_all": clear_all, "clear_tasks": clear_tasks, "clear_facts": clear_facts_only,
}

def get_tool_descriptions() -> str:
    """One-line-per-tool human-readable summary for system prompt."""
    lines = []
    for t in TOOLS:
        params = ", ".join(f"{k}: {v}" for k, v in t["parameters"].items()) or "no params"
        lines.append(f"- {t['name']}({params}) — {t['description']}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(rows, kind="item"):
    if not rows:
        print("No results.")
        return
    for r in rows:
        if kind == "fact":
            print(f"{r['id']}: {r['text']}")
        else:
            when = r.get("when_at") or ""
            status = "done" if r.get("completed") else "open"
            print(f"{r['id']} [{r.get('kind','?')}] {r['title']} ({status}) {when}")

def main():
    import argparse, json
    p = argparse.ArgumentParser(description="Desk Buddy CRUD — all tasks/reminders/facts", prog="crud_tools.py")
    sub = p.add_subparsers(dest="entity", help="what to manage")

    # task subcommands
    t = sub.add_parser("task", help="tasks")
    ts = t.add_subparsers(dest="op")
    tc = ts.add_parser("create", help="create task")
    tc.add_argument("title", help="task title")
    tc.add_argument("--description", default="")
    tc.add_argument("--priority", type=int, default=1)
    tc.add_argument("--when", dest="when_at", default=None, help="due date like 'tomorrow at 5pm' or ISO")
    ts.add_parser("list", help="list tasks").add_argument("--search", default=None, nargs="?", help="filter substring")
    tg = ts.add_parser("get", help="get task by id"); tg.add_argument("id", type=int)
    tu = ts.add_parser("update", help="update task"); tu.add_argument("id", type=int); tu.add_argument("--title", default=None); tu.add_argument("--completed", type=lambda x: x.lower() in ("1","true","yes"), default=None); tu.add_argument("--priority", type=int, default=None)
    td = ts.add_parser("delete", help="delete task"); td.add_argument("id", type=int)
    ts.add_parser("search", help="search tasks").add_argument("query", help="substring")

    # reminder subcommands
    r = sub.add_parser("reminder", help="reminders")
    rs = r.add_subparsers(dest="op")
    rc = rs.add_parser("create", help="create reminder"); rc.add_argument("title"); rc.add_argument("--when", dest="when_at", default=None)
    rs.add_parser("list", help="list reminders").add_argument("--search", default=None, nargs="?", help="filter substring")
    rg = rs.add_parser("get", help="get reminder"); rg.add_argument("id", type=int)
    ru = rs.add_parser("update", help="update reminder"); ru.add_argument("id", type=int); ru.add_argument("--title", default=None); ru.add_argument("--completed", type=lambda x: x.lower() in ("1","true","yes"), default=None)
    rd = rs.add_parser("delete", help="delete reminder"); rd.add_argument("id", type=int)
    rs.add_parser("search", help="search reminders").add_argument("query")

    # fact subcommands
    f = sub.add_parser("fact", help="facts about user")
    fs = f.add_subparsers(dest="op")
    fc = fs.add_parser("create", help="create fact"); fc.add_argument("text", nargs="+")
    fs.add_parser("list", help="list facts").add_argument("--search", default=None, nargs="?", help="filter substring")
    fg = fs.add_parser("get", help="get fact"); fg.add_argument("id", type=int)
    fu = fs.add_parser("update", help="update fact"); fu.add_argument("id", type=int); fu.add_argument("text", nargs="+")
    fd = fs.add_parser("delete", help="delete fact"); fd.add_argument("id", type=int)
    fs.add_parser("search", help="search facts").add_argument("query")

    # item unified
    it = sub.add_parser("item", help="all items (tasks+reminders)")
    its = it.add_subparsers(dest="op")
    its.add_parser("list", help="list all").add_argument("--search", default=None, nargs="?", help="filter substring")
    its.add_parser("search", help="search all").add_argument("query")

    # clear
    c = sub.add_parser("clear", help="clear data")
    c.add_argument("target", nargs="?", default="all", choices=["all","tasks","facts"], help="what to clear")

    # tools manifest
    sub.add_parser("tools", help="print LLM tool manifest (JSON)")

    args = p.parse_args()
    if not args.entity:
        p.print_help()
        # also print human summary
        print("\n--- Available CRUD tools (also importable as `import crud_tools`) ---")
        print(get_tool_descriptions())
        return

    if args.entity == "tools":
        print(json.dumps(TOOLS, indent=2))
        return

    if args.entity == "clear":
        if args.target in ("all",):
            clear_all(); print("Cleared all tasks, reminders, and facts.")
        elif args.target == "tasks":
            clear_tasks(); print("Cleared tasks/reminders.")
        elif args.target == "facts":
            clear_facts_only(); print("Cleared facts.")
        return

    # dispatch
    try:
        if args.entity == "task":
            if args.op == "create":
                r = task_create(args.title, description=args.description, priority=args.priority, when_at=args.when_at)
                print(f"Created task {r['id']}: {r['title']}")
            elif args.op in ("list", None):
                rows = task_list(search=getattr(args, "search", None))
                _print_table(rows)
            elif args.op == "get":
                print(json.dumps(task_get(args.id), indent=2))
            elif args.op == "update":
                upd = {k: v for k, v in {"title": args.title, "priority": args.priority, "completed": args.completed}.items() if v is not None}
                print(json.dumps(task_update(args.id, **upd), indent=2))
            elif args.op == "delete":
                print("Deleted." if task_delete(args.id) else "Not found.")
            elif args.op == "search":
                _print_table(task_search(args.query))
        elif args.entity == "reminder":
            if args.op == "create":
                r = reminder_create(args.title, when_at=args.when_at)
                print(f"Created reminder {r['id']}: {r['title']}")
            elif args.op in ("list", None):
                _print_table(reminder_list(search=getattr(args, "search", None)))
            elif args.op == "get":
                print(json.dumps(reminder_get(args.id), indent=2))
            elif args.op == "update":
                upd = {k: v for k, v in {"title": args.title, "completed": args.completed}.items() if v is not None}
                print(json.dumps(reminder_update(args.id, **upd), indent=2))
            elif args.op == "delete":
                print("Deleted." if reminder_delete(args.id) else "Not found.")
            elif args.op == "search":
                _print_table(reminder_search(args.query))
        elif args.entity == "fact":
            if args.op == "create":
                text = " ".join(args.text)
                fid, created = fact_create(text)
                print(f"{'Created' if created else 'Exists'} fact {fid}: {text}")
            elif args.op in ("list", None):
                rows = fact_list(search=getattr(args, "search", None))
                _print_table(rows, kind="fact")
            elif args.op == "get":
                print(json.dumps(fact_get(args.id), indent=2))
            elif args.op == "update":
                print(json.dumps(fact_update(args.id, " ".join(args.text)), indent=2))
            elif args.op == "delete":
                print("Deleted." if fact_delete(args.id) else "Not found.")
            elif args.op == "search":
                _print_table(fact_search(args.query), kind="fact")
        elif args.entity == "item":
            if args.op in ("list", None, "search"):
                q = getattr(args, "query", None) or getattr(args, "search", None)
                _print_table(item_search(q) if q else item_list())
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
