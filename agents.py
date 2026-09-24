# agents.py - multi-agent with real SQLite CRUD (via crud_tools/db), intent logged
import os, re
from datetime import datetime

# Real DB-backed tools — all CRUD now hits SQLite (facts.db / tasks_reminders.db)
# so .tmp/*.json legacy files are no longer used.
import crud_tools as _crud
from crud_tools import TOOLS, TOOL_FUNCS  # re-export for LLM tool calling

LOG = ".tmp/intent.log"

# --- Task wrappers (backward-compat API, now DB-backed) ---
def task_create(title, desc=""):
    return _crud.task_create(title=title.strip(), description=desc or "")

def task_list():
    return _crud.task_list()

def task_get(tid):
    try:
        return _crud.task_get(int(tid))
    except Exception:
        # fallback: search by title
        for t in _crud.task_list():
            if str(t["id"]) == str(tid) or t["title"].lower() == str(tid).lower():
                return t
        return None

def task_update(tid, title=None, desc=None, status=None):
    try:
        tid_int = int(tid)
    except Exception:
        # title-based lookup
        m = task_get(tid)
        if not m:
            return None
        tid_int = m["id"]
    completed = None
    if status is not None:
        completed = status in ("done", "completed", True, 1)
    return _crud.task_update(tid_int, title=title, description=desc, completed=completed)

def task_delete(tid):
    try:
        return _crud.task_delete(int(tid))
    except Exception:
        m = task_get(tid)
        if not m:
            return False
        return _crud.task_delete(m["id"])

def reminder_create(message, remind_at=None):
    return _crud.reminder_create(title=message.strip(), when_at=remind_at)

def reminder_list():
    return _crud.reminder_list()

def reminder_delete(rid):
    try:
        return _crud.reminder_delete(int(rid))
    except Exception:
        return False

def reminder_update(rid, message=None, remind_at=None):
    try:
        return _crud.reminder_update(int(rid), title=message, when_at=remind_at)
    except Exception:
        return None

# Fact wrappers (new — full CRUD for facts)
def fact_create(text): return _crud.fact_create(text)
def fact_list(): return _crud.fact_list()
def fact_get(fid): return _crud.fact_get(int(fid))
def fact_update(fid, text): return _crud.fact_update(int(fid), text)
def fact_delete(fid): return _crud.fact_delete(int(fid))

def log_intent(prompt, intent, source="llm"):
    os.makedirs(".tmp", exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{datetime.now().isoformat()} | {intent} | {source} | {prompt!r}\n")
    return intent

def classify_intent(prompt, model="gemma4:e2b"):
    low = prompt.lower()
    # fast keyword pre-check for obvious cases (handles offline & speeds up)
    if any(k in low for k in ["remind", "reminder"]):
        return log_intent(prompt, "REMINDER", "keyword")
    if any(k in low for k in ["create task", "add task", "new task", " task "] ) or low.startswith("task"):
        return log_intent(prompt, "TASK", "keyword")
    if low.startswith("remind "):
        return log_intent(prompt, "REMINDER", "keyword")
    try:
        import ollama
        resp = ollama.chat(model=model, messages=[{"role": "user", "content": f"Classify intent as exactly one word: TASK, REMINDER, or CHAT.\nTASK = user wants to manage tasks/todos. REMINDER = user wants a time-based reminder. CHAT = everything else.\nPrompt: {prompt!r}\nReply only TASK, REMINDER, or CHAT."}], options={"num_predict": 5, "num_ctx": 256, "keep_alive": "10m"})
        w = resp.message.content.strip().lower()
        if "reminder" in w:
            return log_intent(prompt, "REMINDER")
        if "task" in w:
            return log_intent(prompt, "TASK")
        return log_intent(prompt, "CHAT")
    except Exception:
        if any(k in low for k in ["remind", "reminder", "remind me", "at ", "tomorrow", "next week"]):
            return log_intent(prompt, "REMINDER", "keyword")
        if any(k in low for k in ["task", "todo", "to-do", "create task", "add task"]):
            return log_intent(prompt, "TASK", "keyword")
        return log_intent(prompt, "CHAT", "keyword")

def _extract_llm_json(prompt, model, instruction):
    try:
        import ollama, json as j
        resp = ollama.chat(model=model, messages=[{"role": "user", "content": instruction + f"\nPrompt: {prompt!r}\nReturn ONLY JSON."}], options={"num_predict": 60, "num_ctx": 256, "keep_alive": "10m"})
        txt = resp.message.content.strip()
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1:
            return j.loads(txt[s:e+1])
    except:
        pass
    return {}

def task_agent(prompt, history, model="gemma4:e2b"):
    data = _extract_llm_json(prompt, model, 'Extract task action as JSON {"action":"create|list|update|delete|get","title":"","description":"","id":"","status":""} . If user says "show tasks" => list. "delete task X" => delete.')
    act = (data.get("action") or "").lower()
    plow = prompt.lower()
    # List before create — handles "tell me my tasks", "what are my tasks", etc.
    is_list = act == "list" or any(k in plow for k in ["list", "show", "what are", "tell me", "my tasks", "my task"])
    if is_list:
        ts = task_list()
        if not ts:
            return "You're all clear — no tasks right now."
        titles = [(t.get('title') or "") for t in ts[-5:]]
        titles = [t for t in titles if t]
        return "Here's what's on your list: " + ", ".join(titles) + "."
    if act == "create" or ("create" in plow or "add task" in plow):
        title = data.get("title") or re.sub(r"(?i)create task|add task", "", prompt).strip() or prompt.strip()
        t = task_create(title, data.get("description") or "")
        return f"Got it — added '{t['title']}' to your list."
    if act == "delete":
        ok = task_delete(data.get("id") or data.get("title") or prompt)
        return "Done — removed that one for you." if ok else "Hmm, I couldn't find that task."
    if act == "update":
        t = task_update(data.get("id") or data.get("title") or "", title=data.get("title"), status=data.get("status"))
        return f"Updated '{t['title']}' for you." if t else "Hmm, I couldn't find that one to update."
    if len(prompt.split()) <= 8:
        t = task_create(prompt)
        return f"Got it — added '{t['title']}' to your list."
    return "Just say something like 'add buy milk to my list' or 'show my tasks' and I'll handle it."

def reminder_agent(prompt, history, model="gemma4:e2b"):
    data = _extract_llm_json(prompt, model, 'Extract reminder as JSON {"action":"create|list|delete","message":"","remind_at":"ISO or natural like tomorrow 5pm","id":""}')
    act = (data.get("action") or "").lower()
    plow = prompt.lower()
    # List intent first — must beat create. Handles "tell me my reminders", "what are my reminders", etc.
    is_list = act == "list" or any(k in plow for k in ["list", "show", "what are", "tell me", "my reminders", "my reminder"])
    if is_list:
        rs = reminder_list()
        if not rs:
            return "No reminders at the moment — you're all clear."
        # DB dict uses 'title', legacy used 'message' — support both
        msgs = [(r.get('title') or r.get('message') or "") for r in rs[-5:]]
        msgs = [m for m in msgs if m]
        if not msgs:
            return "No reminders at the moment — you're all clear."
        return "Here are your reminders: " + ", ".join(msgs) + "."
    if act == "delete":
        ok = reminder_delete(data.get("id") or "")
        return "Done — removed that reminder." if ok else "Hmm, couldn't find that one."
    if act == "create" or "remind" in plow:
        raw_msg = data.get("message") or data.get("title") or prompt
        r = reminder_create(raw_msg, data.get("remind_at") or data.get("when_at"))
        title = r.get('title') or r.get('message') or raw_msg
        when = r.get('when_at') or r.get('remind_at') or ''
        # friendly when if ISO
        try:
            if when:
                from memory_handler import friendly_when
                when = friendly_when(when)
        except Exception:
            pass
        return f"Got it — I'll remind you about '{title}'" + (f" {when}." if when else ".")
    # fallback: if no action matched but we got a list-like prompt without keyword, treat as list
    if any(k in plow for k in ["remind", "reminder"]):
        rs = reminder_list()
        if rs:
            msgs = [(r.get('title') or r.get('message') or "") for r in rs[-5:]]
            return "Here are your reminders: " + ", ".join(m for m in msgs if m) + "."
    r = reminder_create(prompt)
    title = r.get('title') or r.get('message') or prompt
    return f"Got it — I'll remind you about '{title}'."

def chat_agent(prompt, history, model="gemma4:e2b"):
    import ollama
    sh = ([history[0]] + history[-10:]) if history and history[0].get("role") == "system" else history[-10:]
    resp = ollama.chat(model=model, messages=sh + [{"role": "user", "content": prompt}], options={"num_predict": 32, "num_ctx": 256, "keep_alive": "10m"})
    return resp.message.content.strip()
