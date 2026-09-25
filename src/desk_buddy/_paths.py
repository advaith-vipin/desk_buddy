"""Shared path resolution for the Desk Buddy package.

Everything runtime-ish (databases, .tmp, venv) lives at the *project root*
— the directory containing ``run.sh`` — so data survives refactors and the
package stays relocatable.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_project_root(start: str) -> str:
    d = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(d, "run.sh")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(PACKAGE_DIR)  # fallback: src/


ROOT = _find_project_root(PACKAGE_DIR)
SRC_DIR = os.path.dirname(PACKAGE_DIR)
TMP_DIR = os.path.join(ROOT, ".tmp")

FACTS_DB_PATH = os.path.join(ROOT, "facts.db")
TASKS_DB_PATH = os.path.join(ROOT, "tasks_reminders.db")


def ensure_tmp_dir() -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    return TMP_DIR