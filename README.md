# desk_buddy

A live lip-syncing desk avatar (pytoon + pygame) with an Ollama-powered chat
memory: tasks, reminders, and user facts stored in SQLite.

## Layout

```
src/desk_buddy/      Python package (all source)
  entrypoints/       Runnable apps: pytoon_live.py (live player, default),
                     pytoon_cli.py (one-shot video render)
  core/              Data + logic: db.py (SQLite), memory_handler.py (facts/NL),
                     agents.py (intent + task/reminder/chat agents),
                     crud_tools.py (tool layer over db)
  avatar/            Renderers: cube_face.py (cube-bot), front_face.py (cartoon)
  _paths.py          Runtime path anchors (ROOT, TMP_DIR, db paths)
  prompts.py         All LLM system prompts / instruction templates
scripts/             Example + training scripts
run.sh               Launcher (interactive by default)
patch_pytoon.py      Post-install pytoon -> moviepy 2.x compatibility patch
```

## Quick start

```bash
./run.sh                 # interactive launcher (live echo / chat / CLI / setup)
./run.sh --chat --fast   # chat mode directly, skipping prompts
./run.sh --cli --text "Hello"   # one-shot video render
./run.sh --setup         # create venv, install deps, patch pytoon
./run.sh --help          # options for the target entry point
```

Equivalently, without the wrapper:

```bash
PYTHONPATH=src .venv/bin/python -m desk_buddy.entrypoints.pytoon_live --chat
PYTHONPATH=src .venv/bin/python -m desk_buddy.entrypoints.pytoon_cli --text "Hello"
```

## Data

Runtime state lives at the project root (git-ignored):

- `.tmp/` — speech renders, chat history, MEMORY.md
- `facts.db`, `tasks_reminders.db` — long-term memory