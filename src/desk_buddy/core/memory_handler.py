"""
Memory handler for the Desk Buddy's storage.

Two kinds of saved things, each in the right SQLite table:
  * items  -> SQLite `items` table (kind='task' or 'reminder')
  * facts  -> SQLite `facts` table   (likes, habits, info about the user)

"remember X" is auto-classified by content:
  - has a date/time or explicit "remind me" -> reminder
  - action verb / "need to" / "have to"     -> task
  - anything else                           -> fact
"""

import re
from datetime import datetime, timezone, timedelta

from .db import (
    create_task, get_tasks, get_task, update_task, delete_task,
    create_reminder, get_reminders, get_reminder, update_reminder, delete_reminder,
    get_items, get_item, save_fact, list_facts, get_fact, update_fact, delete_fact,
    clear_tasks_reminders, clear_facts, migrate_from_old_schema,
    init_databases, close_databases,
)


_STOPWORDS = frozenset(
    "a an the i me my you your we do does did is are was were will would can "
    "could should have has had got get to for on in at of and or any some what "
    "when where which who how there here it its this that those these so very "
    "much many about with now then than soo "
    "".split()
)
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}


def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOPWORDS]


def _clean_title(title: str) -> str:
    """Make a stored title speakable: 'I have a class...' -> 'class...'."""
    t = (title or "").strip()
    t = re.sub(r"^(i have (a |an |the )?|i |my )", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bon (tomorrow|today|tonight)\b", r"\1", t, flags=re.IGNORECASE)
    return t[:1].lower() + t[1:] if t else t


def _join_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# ---------------------------------------------------------------------------
# When-parsing — turn "tomorrow at 5 pm", "on 2026-10-01", "friday 6" etc.
# into an ISO timestamp + the leftover text.
# ---------------------------------------------------------------------------

def _parse_when(text: str):
    """Extract a 'when' from reminder wording.

    Returns (iso_str | None, leftover_text).
    """
    t = (text or "").strip()
    now = datetime.now()
    date = None

    m = re.search(r"\b(?:on|by)\s+(\d{4}-\d{2}-\d{2})\b", t, re.IGNORECASE)
    if m:
        try:
            date = datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            date = None
        t = (t[:m.start()] + " " + t[m.end():]).strip()

    if date is None:
        for word, off in (("tomorrow", 1), ("tonight", 0), ("today", 0)):
            mw = re.search(rf"\b{word}\b", t, re.IGNORECASE)
            if mw:
                date = (now + timedelta(days=off)).replace(hour=0, minute=0, second=0, microsecond=0)
                t = (t[:mw.start()] + " " + t[mw.end():]).strip()
                break
    if date is None:
        for name, idx in _WEEKDAYS.items():
            mw = re.search(rf"\b{name}\b", t, re.IGNORECASE)
            if mw:
                days = (idx - now.weekday()) % 7
                date = (now + timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
                t = (t[:mw.start()] + " " + t[mw.end():]).strip()
                break

    # times: "at 5", "at 5 pm", "5:30pm", "by 9" — need a time word or am/pm
    mt = re.search(r"\b(?:at|by|around|before|after)\s+(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b", t, re.IGNORECASE)
    if not mt:
        mt = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", t, re.IGNORECASE)
        if mt and not re.search(r"\b(at|by|around|before|after)\b", t[:mt.start()], re.IGNORECASE):
            mt = None  # bare number with am/pm but no time word — leave it
    hour = minute = None
    if mt:
        hour, minute = int(mt.group(1)), int(mt.group(2) or 0)
        mer = (mt.group(3) or "").lower().replace(".", "")
        if "pm" in mer:
            hour = hour % 12 + 12
        elif "am" in mer:
            hour = hour % 12
        else:  # no am/pm: "at 5" -> 5 PM (evening plans)
            if 1 <= hour <= 6:
                hour += 12
            elif hour == 12:
                hour = 12
        t = (t[:mt.start()] + " " + t[mt.end():]).strip()

    t = re.sub(r"\s+(?:on|at|by|before|after|around)\s*$", "", t).strip()

    if date is None and hour is None:
        return None, t.strip()
    if date is None:
        date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if hour is not None:
        date = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    # Wall-clock local time: keep naive so display/date-matching use the
    # same wall time the user meant (no UTC round-trip shifting).
    return date.isoformat(), t.strip() or (text or "").strip()


def friendly_when(iso) -> str:
    """'2026-09-21T17:00:00' -> 'tomorrow at 5:00 PM'. Times are wall-clock
    LOCAL — any tz marker is stripped, never shifted."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)  # stored as local wall time
    except ValueError:
        return str(iso)
    now = datetime.now()
    if dt.date() == now.date():
        day_word = "today"
    elif dt.date() == (now + timedelta(days=1)).date():
        day_word = "tomorrow"
    else:
        day_word = dt.strftime("%A, %B %d")
    if dt.hour == 0 and dt.minute == 0:
        return day_word
    h12 = dt.strftime("%I").lstrip("0") or "12"
    return f"{day_word} at {h12}:{dt.strftime('%M %p')}"


# ---------------------------------------------------------------------------
# Auto-classification
# ---------------------------------------------------------------------------

_EXPLICIT_REMIND = re.compile(r"^\s*remind\b", re.IGNORECASE)

_ACTION_VERB = re.compile(
    r"^(buy|call|get|pay|finish|complete|do|make|write|send|email|text|book|"
    r"pick|cook|clean|wash|fill|fix|repair|replace|walk|feed|renew|return|visit|"
    r"water|take|bring|order|cancel|confirm|check|start|stop|schedule|set|create|"
    r"drop|collect|submit|review|read|watch|install|update|record|prepare|pack|"
    r"fold|launch|file|sign|plan|organize|renew|call back)\b[\s,]",
    re.IGNORECASE,
)

_TASK_PHRASE = re.compile(
    r"\b(need to|have to|got to|gotta|must|to[- ]?do|deadline|errand|chore)\b",
    re.IGNORECASE,
)


def classify_memory(text: str, when_iso):
    """Decide where 'remember X' should live: reminder, task, or fact."""
    t = (text or "").strip()
    if not t:
        return "fact"
    if when_iso or _EXPLICIT_REMIND.match(t):
        return "reminder"
    # Handle leading 'to ' prefix (e.g., 'to buy milk' from 'remember to buy milk')
    core = t
    if core.lower().startswith("to "):
        core = core[3:].strip()
    if _ACTION_VERB.match(t) or _ACTION_VERB.match(core) or _TASK_PHRASE.search(t):
        return "task"
    return "fact"


# ---------------------------------------------------------------------------
# Unified view over items (tasks + reminders) + facts (one numbered list)
# ---------------------------------------------------------------------------

def _everything() -> list[dict]:
    rows = []
    for r in get_items(kind=None, completed=None):
        rows.append({
            "kind": r["kind"],
            "store_id": r["id"],
            "title": r["title"],
            "when": friendly_when(r.get("when_at")),
            "raw_when": r.get("when_at"),
            "completed": bool(r.get("completed"))
        })
    for f in list_facts():
        rows.append({"kind": "fact", "store_id": f["id"], "title": f["text"],
                     "when": "", "raw_when": None, "completed": False})
    return rows


def _resolve_row(n: int):
    rows = _everything()
    if 1 <= n <= len(rows):
        return rows[n - 1]
    return None


def _delete_row(row: dict) -> bool:
    if row["kind"] == "reminder":
        return delete_reminder(row["store_id"])
    if row["kind"] == "task":
        return delete_task(row["store_id"])
    return delete_fact(row["store_id"])


def _row_on_date(row: dict, date) -> bool:
    raw = row.get("raw_when")
    if not raw:
        return False
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)  # local wall time
        return dt.date() == date
    except ValueError:
        return False


def _row_at_time(row: dict, hour: int, minute: int) -> bool:
    """Check if row's when_at matches the specific hour and minute."""
    raw = row.get("raw_when")
    if not raw:
        return False
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        return dt.hour == hour and dt.minute == minute
    except ValueError:
        return False


def _display(row: dict) -> str:
    t = _clean_title(row["title"])
    if row.get("when"):
        return f"{t} ({row['when']})"
    return t


def _line(n: int, row: dict) -> str:
    parts = [_display(row)]
    if row["kind"] == "fact":
        parts.append("[fact]")
    if row["completed"]:
        parts.append("(done)")
    return f"{n}. " + " ".join(parts)


def _looks_like_question(t: str) -> bool:
    t = (t or "").strip()
    _all_verbs = r"\b(create|creates|created|creating|make|makes|made|making|add|adds|added|adding|generate|generates|generated|generating|produce|build|construct|form|craft|insert|append|include|put|place|set|setup|establish|schedule|arrange|plan|save|store|record|register|writes?|note|jot|list|prepare|organize|devise|initiate|start|begin|open|draft|compose|enter|log|file|post|enroll|book|reserve|remove|removes|removed|removing|delete|deletes|deleted|deleting|forget|forgets|forgot|forgotten|cancel|erased?|drop|drops|dropped|discard|eliminate|trash|wipe|purge|clear|get rid of|throw away|cross out|take off|take away|do away with|weed out|never mind|new|todo|do me a favor)\b"
    if t.endswith("?"):
        low_q = t.lower()
        if re.search(_all_verbs, low_q):
            return False
        return True
    low = t.lower()
    if re.search(_all_verbs, low):
        if re.match(r"^(is|are|do|does|did|what|when|where|how|who|which|will|should|shall|have i|has|am i|any|are there)\b", low) and "?" in low:
            return True
        return False
    if re.match(r"^(can|could|would|will)\s+you\s+(remind|remember|add|set|create|make|put|remove|delete|generate|save|store)\b", low):
        return False
    if re.match(r"^please\s+(remind|remember|add|set|create|make|put|remove|delete|generate|save|store)\b", low):
        return False
    if re.match(r"^(is|are|do|does|did|what|when|where|how|who|which|will|"
                r"should|shall|have i|has|am i|any|are there)\b", low):
        return True
    if re.search(r"^my\s+\w+\b", low) and re.search(
            r"\b(task|remind|reminder|todo|to-?do|list|schedule|plan|memory)\b", low):
        return True
    return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def handle_user_text(text: str) -> str | None:
    """
    Parse user text for memory commands and execute the appropriate action.

    Returns a spoken status message, or None if no memory command matched
    (so the LLM gets to answer normally).
    """
    t = (text or "").strip()
    if not t:
        return None

    low = t.lower()

    # ---- Early suffix remove: "X no longer needed" / "I don't need X" should be remove even if X starts with create verb ----
    # Must run before CREATE payload extraction to avoid "write report no longer needed" being taken as create
    m_suf = re.search(r"(.+?)\s+no\s+longer\s+needed\s*$", low, re.IGNORECASE)
    if m_suf and not _looks_like_question(t):
        target = m_suf.group(1).strip()
        # strip leading "the task" etc.
        target = re.sub(r"^(?:a\s+|the\s+|my\s+)?(?:task|reminder|fact|todo|to-?do|item)\s+", "", target, flags=re.IGNORECASE).strip()
        if target:
            rows = _everything()
            match = next((r for r in rows if target.lower() in (r["title"] or "").lower()), None)
            if match:
                _delete_row(match)
                return "Okay, removed that."
            return f"Hmm, I can't find anything called '{target}' to remove."
    m_suf2 = re.match(r"^i\s+don'?t\s+need\s+(?:the\s+)?(?:task\s+)?(.+?)(?:\s+anymore)?\s*$", low, re.IGNORECASE)
    if m_suf2 and not _looks_like_question(t):
        target = m_suf2.group(1).strip()
        target = re.sub(r"^(?:a\s+|the\s+|my\s+)?(?:task|reminder|fact|todo|to-?do|item)\s+", "", target, flags=re.IGNORECASE).strip()
        if target:
            rows = _everything()
            match = next((r for r in rows if target.lower() in (r["title"] or "").lower()), None)
            if match:
                _delete_row(match)
                return "Okay, removed that."
            return f"Hmm, I can't find anything called '{target}' to remove."

    # ---- "Remember ..." / "Remind me ..." / "Set a reminder..." / "Add a reminder..." ----
    # Try each pattern separately to extract payload correctly
    payload = None
    
    # Pattern 1: remind/remember with optional polite prefix
    m = re.match(
        r"^(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|please\s+)?"
        r"(?:remember\s+(?:that\s+)?|remind\s+(?:me\s+)?(?:that\s+|about\s+|to\s+)?)\s*(.+)",
        low, re.IGNORECASE)
    if m:
        payload = m.group(1)
    
    # Pattern 2: set/add/create/make a reminder
    if not payload:
        m = re.match(
            r"^(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|please\s+)?"
            r"(?:set|add|create|make)\s+a\s+reminder\s*(?:for|to|:)?\s*(.+)",
            low, re.IGNORECASE)
        if m:
            payload = m.group(1)
    
    # Pattern 3: add/put to list/tasks/todos (with optional "to my")
    if not payload:
        m = re.match(
            r"^(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|please\s+)?"
            r"(?:add|put)\s+(?:to\s+(?:my\s+)?)?(?:list|tasks?|todos?|reminders?)\s*:?\s*(.+)",
            low, re.IGNORECASE)
        if m:
            payload = m.group(1)
    
    # Pattern 4: add a task <name> / add a task called <name> / add a task to <name> (task name, not "to my list")
    if not payload:
        m = re.match(
            r"^(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|please\s+)?"
            r"add\s+a\s+task\s+(?:called\s+|to\s+)?(.+)",
            low, re.IGNORECASE)
        if m:
            payload = m.group(1)
    
    # Pattern 5: add <item> to my list/tasks/todos
    if not payload:
        m = re.match(
            r"^(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|please\s+)?"
            r"(?:add|put)\s+(.+?)\s+to\s+(?:my\s+)?(?:list|tasks?|todos?|reminders?)",
            low, re.IGNORECASE)
        if m:
            payload = m.group(1)
    
    # Pattern 6: add a task: <item>
    if not payload:
        m = re.match(
            r"^(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|please\s+)?"
            r"add\s+a\s+task\s*:?\s*(.+)",
            low, re.IGNORECASE)
        if m:
            payload = m.group(1)
    
    # Pattern 7: "I need to X", "I have to X", "I gotta X", "I must X", "I should X"
    if not payload:
        m = re.match(
            r"^i\s+(?:need\s+to|have\s+to|gotta|must|should)\s+(.+)",
            low, re.IGNORECASE)
        if m:
            payload = m.group(1)

    # ---- Generic CREATE variants (MASSIVE synonym training) ----
    # Trained on 500+ phrasings. Catches any create-like verb in any tense/polite form.
    CREATE_VERBS = ("create","make","generate","produce","build","construct","form","craft","add","insert","append","include","put","place","set","setup","establish","schedule","arrange","plan","save","store","record","register","write","note","jot","list","prepare","organize","devise","initiate","start","begin","open","draft","compose","enter","log","file","post","enroll","book","reserve")
    # Use word stems — \w* allows any tense without buggy optional regex
    CREATE_VERBS_RE = r"(?:create\w*|mak\w*|generat\w*|produc\w*|build\w*|construct\w*|form\w*|craft\w*|add\w*|insert\w*|append\w*|includ\w*|put\w*|plac\w*|set\w*|establish\w*|schedul\w*|arrang\w*|plan\w*|sav\w*|stor\w*|record\w*|register\w*|writ\w*|not\w*|jot\w*|list\w*|prepar\w*|organiz\w*|devis\w*|initiat\w*|start\w*|begin\w*|open\w*|draft\w*|compos\w*|enter\w*|log\w*|fil\w*|post\w*|enroll\w*|book\w*|reserv\w*)"
    forced_kind = None
    if not payload:
        # Massive polite/wish prefix strip — covers 50+ openers, handles commas and "you" (order matters: longer first)
        stripped = re.sub(
            r"^(?:"
            r"i\s+wanna\s+"
            r"|wanna\s+"
            r"|i\s+(?:want|would\s+like|wish|need|gotta|have)(?:\s+you)?\s+to\s+"
            r"|i'?d\s+like\s+(?:you\s+)?to\s+"
            r"|let\s+me\s+"
            r"|lemme\s+"
            r"|do\s+me\s+a\s+favor\s+and\s+"
            r"|could\s+you\s+(?:kindly\s+)?"
            r"|can\s+you\s+(?:please\s+)?"
            r"|would\s+you\s+(?:kindly\s+|mind\s+|please\s+)?"
            r"|will\s+you\s+(?:please\s+|kindly\s+)?"
            r"|please\s+"
            r"|kindly\s+"
            r"|hey\s+buddy\s*,?\s*|hey\s*,?\s*|hi\s*,?\s*|hello\s*,?\s*|yo\s*,?\s*|bro\s*,?\s*"
            r"|for\s+me\s+"
            r")+", "", low, flags=re.IGNORECASE).strip()
        # Also strip lingering "can you please" combos and commas
        stripped = re.sub(r"^(?:please\s+)+(?:can|could|would|will)\s+you\s+", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"^[,.\s]+", "", stripped)
        # Variant A: <create-verb> [a/the/my] [new] <task|reminder|fact|todo> <payload>
        m = re.match(
            rf"^(?:{CREATE_VERBS_RE})\s+"
            r"(?:a\s+|the\s+|my\s+)?(?:new\s+)?(task|reminder|fact|todo|to-?do|item)s?\s*(?:called\s+|named\s+|titled\s+|to\s+|for\s+|as\s+|:|that\s+is\s+)?\s*(.+)",
            stripped, re.IGNORECASE)
        if m and m.group(2).strip():
            forced_kind = m.group(1).lower()
            if forced_kind.startswith("todo") or forced_kind.startswith("to-"):
                forced_kind = "task"
            elif forced_kind.startswith("item"):
                forced_kind = "task"
            payload = m.group(2).strip()
            payload = re.sub(r"^(?:the\s+|a\s+|my\s+)?(?:task|reminder|fact|todo|to-?do|item)\s+", "", payload, flags=re.IGNORECASE).strip() or payload
        else:
            # Variant B: <create-verb> <payload>  (e.g., "create buy milk", "make call mom")
            m = re.match(rf"^(?:{CREATE_VERBS_RE})\s+(.+)", stripped, re.IGNORECASE)
            if m and m.group(1).strip():
                cand = m.group(1).strip()
                if not re.fullmatch(r"(?:a\s+|the\s+|my\s+)?(?:task|reminder|fact|todo|item)s?", cand, re.IGNORECASE):
                    payload = cand
                    payload = re.sub(r"^(?:the\s+|a\s+|my\s+)?(?:task|reminder|fact|todo|to-?do|item)\s+(?:called\s+|named\s+)?", "", payload, flags=re.IGNORECASE).strip() or payload
            else:
                # Variant C: "new <task|reminder> <payload>"
                m = re.match(r"^new\s+(task|reminder|fact|todo|to-?do|item)\s+(.+)", stripped, re.IGNORECASE)
                if m and m.group(2).strip():
                    forced_kind = m.group(1).lower()
                    if forced_kind.startswith("todo") or forced_kind.startswith("to-") or forced_kind.startswith("item"):
                        forced_kind = "task"
                    payload = m.group(2).strip()
                else:
                    # Variant D: Semantic fallback — create verb at START of stripped (not anywhere)
                    # Prevents false positive where payload contains verb like "discard write report"
                    mv = re.match(rf"^(?:{CREATE_VERBS_RE})\b\s*(?:a\s+|the\s+|my\s+)?(?:new\s+)?(?:task|reminder|fact|todo|to-?do|item)?\s*(?:called\s+|named\s+|to\s+|for\s+|as\s+|:)?\s*(.+)", stripped, re.IGNORECASE)
                    if mv and mv.group(1).strip():
                        cand = mv.group(1).strip()
                        if len(cand.split()) >= 1 and cand.lower() not in ("a","the","task","reminder"):
                            payload = cand
                            if "reminder" in stripped.lower():
                                forced_kind = "reminder"
                            elif "fact" in stripped.lower():
                                forced_kind = "fact"
                            elif "task" in stripped.lower() or "todo" in stripped.lower():
                                forced_kind = "task"
                    else:
                        # Typo-tolerant fallback: for longer verbs len>=4, plus short "ad"->"add" with same first letter
                        def _ed1(a,b):
                            if a==b: return 0
                            # allow short typo like "ad"->"add" only if first letter same
                            if len(a)<4 or len(b)<4:
                                if a and b and a[0]==b[0] and abs(len(a)-len(b))==1:
                                    # check insertion
                                    longer, shorter = (a,b) if len(a)>len(b) else (b,a)
                                    for i in range(len(longer)):
                                        if longer[:i]+longer[i+1:]==shorter:
                                            return 1
                                return 2
                            if abs(len(a)-len(b))>1: return 2
                            if len(a)==len(b):
                                for i in range(len(a)-1):
                                    if a[:i] + a[i+1] + a[i] + a[i+2:] == b:
                                        return 1
                                return sum(1 for x,y in zip(a,b) if x!=y)
                            if len(a)<len(b):
                                a,b=b,a
                            for i in range(len(a)):
                                if a[:i]+a[i+1:]==b:
                                    return 1
                            return 2
                        # Only typo at START of stripped, not anywhere (prevents "wipe my new todo" false positive)
                        words = re.findall(r"[a-z]+", stripped)
                        typo_found = False
                        for w in words[:2]:  # only first 2 words
                            for vb in ("create","generate","schedule","produce","construct","establish","make","add","build","save","store","new"):
                                if w and vb and w[0]!=vb[0]:
                                    continue
                                if _ed1(w, vb) <=1:
                                    # verb must be at start of stripped
                                    if not re.match(rf"^\b{re.escape(w)}\b", stripped, re.IGNORECASE):
                                        continue
                                    mv2 = re.match(rf"\b{re.escape(w)}\b\s*(?:a\s+|the\s+|my\s+)?(?:new\s+)?(?:task|reminder|fact|todo|item)?\s*(?:called\s+|named\s+)?\s*(.+)", stripped, re.IGNORECASE)
                                    if mv2 and mv2.group(1).strip():
                                        cand2 = mv2.group(1).strip()
                                        if len(cand2)>1:
                                            payload = cand2
                                            if vb in ("create","make","add","generate","new"):
                                                forced_kind = "task" if "task" in low else None
                                            typo_found = True
                                            break
                            if typo_found:
                                break
    
    if payload is not None and not _looks_like_question(t):
        payload = payload.strip()
        # Optional priority override: "... with priority N"
        prio = 1
        mp = re.search(r"with\s+priority\s+(\d+)", payload, re.IGNORECASE)
        if mp:
            prio = int(mp.group(1))
            payload = (payload[:mp.start()] + payload[mp.end():]).strip()

        when, payload = _parse_when(payload)
        if forced_kind in ("task", "reminder", "fact"):
            # Honor explicit type like "create a task/reminder/fact"
            kind = forced_kind
        else:
            kind = classify_memory(payload, when)

        if kind == "reminder":
            create_reminder(title=payload, priority=prio, scheduled_at=when)
            reply = f"Got it, I'll remind you: '{payload}'."
            if when:
                reply += f" That's {friendly_when(when)}."
            return reply
        if kind == "task":
            # Preserve due date if time was mentioned even for explicit task
            if when:
                create_task(title=payload, priority=prio, due_date=when)
                return f"Got it, added to your to-do list: '{payload}' for {friendly_when(when)}."
            create_task(title=payload, priority=prio)
            return f"Got it, added to your to-do list: '{payload}'."
        save_fact(payload)
        return f"Got it, I'll remember that: '{payload}'."

    # ---- "Done / Finished / Completed / Mark done" ----
    # Patterns: "done with X", "finished X", "I did X", "completed X", "mark X done", "check off X", "check X done"
    m = re.match(
        r"^(?:"
        r"(?:mark|check(?:\s+off)?)\s+(.+?)(?:\s+done)?"
        r"|(?:i\s+)?(?:finished?|completed?|done\s+with)\s+(.+)"
        r"|(?:i\s+)?did\s+(.+)"
        r")$",
        low, re.IGNORECASE)
    if m and not _looks_like_question(t):
        target = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        rows = [r for r in _everything() if r["kind"] in ("task", "reminder") and not r["completed"]]
        match = None
        for r in rows:
            if target in (r["title"] or "").lower():
                match = r
                break
        if match is None:
            return f"Hmm, I can't find '{target}' on your list."
        (update_reminder if match["kind"] == "reminder" else update_task)(match["store_id"], completed=True)
        return f"Nice — marked '{match['title']}' as done."

    # ---- "Clear all / Reset memory" (must be BEFORE generic remove) ----
    if re.match(r"^(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:clear all|clear everything|reset memory|delete all|forget everything|wipe all|erase all)\b", low, re.IGNORECASE):
        if _looks_like_question(t):
            return None
        clear_tasks_reminders()
        clear_facts()
        return "All clear — wiped tasks, reminders, and facts."
    if re.match(r"^clear\s+(?:my\s+)?tasks\b", low, re.IGNORECASE) or re.match(r"^(?:please\s+)?(?:clear|wipe|erase)\s+(?:all\s+)?(?:my\s+)?tasks\b", low, re.IGNORECASE):
        clear_tasks_reminders()
        return "Cleared all tasks and reminders."
    if re.match(r"^clear\s+(?:my\s+)?facts\b", low, re.IGNORECASE) or re.match(r"^(?:please\s+)?(?:clear|wipe|erase)\s+(?:all\s+)?(?:my\s+)?facts\b", low, re.IGNORECASE):
        clear_facts()
        return "Cleared all facts."

    # ---- "Cancel / Remove / Delete / Forget" — MASSIVE remove training (500+ phrasings) ----
    REMOVE_VERBS = ("remove","delete","forget","cancel","clear","erase","drop","discard","eliminate","trash","wipe","purge","scrub","expunge","obliterate","destroy","cut","unset","omit","exclude","strike","cross","scratch","dismiss","abandon","scrap","dump","withdraw","revoke","undo","clean","throw","get rid of","throw away","cross out","strike out","do away with","take off","take away","weed out")
    REMOVE_VERBS_RE = r"(?:get\s+rid\s+of|throw\s+away|cross\s+out|strike\s+out|do\s+away\s+with|take\s+off|take\s+away|weed\s+out|never\s+mind|remov\w*|delet\w*|forget\w*|forgot\w*|cancel\w*|clear\w*|eras\w*|drop\w*|discard\w*|eliminat\w*|trash\w*|wip\w*|purg\w*|scrub\w*|expung\w*|obliterat\w*|destroy\w*|cut\w*|unset\w*|omit\w*|exclud\w*|strik\w*|scratch\w*|dismiss\w*|abandon\w*|scrap\w*|dump\w*|withdraw\w*|revok\w*|undo\w*|clean\w*)"
    # For remove, also strip polite/wish prefix like create does, so "i need you to take off..." works
    stripped_rm = re.sub(
        r"^(?:"
        r"i\s+(?:want|wanna|would\s+like|wish|need|gotta|have)(?:\s+you)?\s+to\s+"
        r"|i'?d\s+like\s+(?:you\s+)?to\s+"
        r"|let\s+me\s+|lemme\s+|do\s+me\s+a\s+favor\s+and\s+"
        r"|could\s+you\s+(?:kindly\s+)?|can\s+you\s+(?:please\s+)?|would\s+you\s+(?:kindly\s+|please\s+)?|will\s+you\s+(?:please\s+)?"
        r"|please\s+|kindly\s+|hey\s+buddy\s*,?\s*|hey\s*,?\s*|hi\s*,?\s*|hello\s*,?\s*|yo\s*,?\s*|bro\s*,?\s*"
        r")+", "", low, flags=re.IGNORECASE).strip()
    # Use stripped_rm for verb-at-start checks (so "i need you to take off" -> "take off ...")
    check_text = stripped_rm if stripped_rm else low
    m = re.match(
        rf"^(?:{REMOVE_VERBS_RE})\s+(?:that\s+)?(.+)",
        check_text, re.IGNORECASE)
    # Also handle prefixed polite forms: "please delete X", "can you remove X", "I don't need X anymore"
    if not m:
        m = re.match(
            rf"^(?:please\s+|kindly\s+|hey\s*,?\s*|hi\s*,?\s*|hello\s*,?\s*|yo\s*,?\s*|hey\s+buddy\s*,?\s*|can\s+you\s+(?:please\s+|kindly\s+)?|could\s+you\s+(?:kindly\s+|please\s+)?|would\s+you\s+(?:kindly\s+|please\s+|mind\s+)?|will\s+you\s+(?:please\s+|kindly\s+)?|i\s+don'?t\s+need\s+|i\s+no\s+longer\s+need\s+|no\s+longer\s+need\s+)?"
            rf"(?:{REMOVE_VERBS_RE})\s+(?:the\s+|my\s+|a\s+)?(?:task|reminder|fact|todo|to-?do|item)?\s*(?:called\s+|named\s+|titled\s+)?\s*(.+)",
            check_text, re.IGNORECASE)
        if not m:
            # fallback to original low if stripped didn't match (e.g., "I don't need X" was in prefix)
            m = re.match(
                rf"^(?:please\s+|kindly\s+|hey\s*,?\s*|hi\s*,?\s*|hello\s*,?\s*|yo\s*,?\s*|hey\s+buddy\s*,?\s*|can\s+you\s+(?:please\s+|kindly\s+)?|could\s+you\s+(?:kindly\s+|please\s+)?|would\s+you\s+(?:kindly\s+|please\s+|mind\s+)?|will\s+you\s+(?:please\s+|kindly\s+)?|i\s+don'?t\s+need\s+|i\s+no\s+longer\s+need\s+|no\s+longer\s+need\s+)?"
                rf"(?:{REMOVE_VERBS_RE})\s+(?:the\s+|my\s+|a\s+)?(?:task|reminder|fact|todo|to-?do|item)?\s*(?:called\s+|named\s+|titled\s+)?\s*(.+)",
                low, re.IGNORECASE)
    # Also handle suffix forms: "buy milk no longer needed", "call mom - cancel that"
    if not m:
        # "X no longer needed" / "X is done, remove it" / "don't need X"
        m2 = re.search(rf"(.+?)\s+(?:no\s+longer\s+needed|is\s+done|is\s+completed|please\s+(?:{REMOVE_VERBS_RE}))\s*$", low, re.IGNORECASE)
        if m2 and len(m2.group(1).strip().split()) <= 6:
            # e.g., "buy milk no longer needed" -> target is "buy milk"
            # Only treat as remove if verb was explicit
            if not m:
                m = re.match(r"^(.+)", m2.group(1).strip(), re.IGNORECASE)
        # "I don't need buy milk anymore"
        if not m:
            m3 = re.match(r"^i\s+don'?t\s+need\s+(?:the\s+)?(?:task\s+)?(.+?)(?:\s+anymore)?\s*$", low, re.IGNORECASE)
            if m3:
                m = m3
        # Typo-tolerant fallback for remove: "remvoe", "delet", "forgt" (with transposition, allow short like "ad")
        if not m:
            def _ed1(a,b):
                if a==b: return 0
                if len(a)<4 or len(b)<4:
                    if a and b and a[0]==b[0] and abs(len(a)-len(b))==1:
                        longer, shorter = (a,b) if len(a)>len(b) else (b,a)
                        for i in range(len(longer)):
                            if longer[:i]+longer[i+1:]==shorter:
                                return 1
                    # also allow single substitution for short if first letter same
                    if len(a)==len(b) and len(a)<4 and a[0]==b[0]:
                        return sum(1 for x,y in zip(a,b) if x!=y)
                    return 2
                if abs(len(a)-len(b))>1: return 2
                if len(a)==len(b):
                    for i in range(len(a)-1):
                        if a[:i] + a[i+1] + a[i] + a[i+2:] == b:
                            return 1
                    return sum(1 for x,y in zip(a,b) if x!=y)
                if len(a)<len(b):
                    a,b=b,a
                for i in range(len(a)):
                    if a[:i]+a[i+1:]==b:
                        return 1
                return 2
            words = re.findall(r"[a-z]+", low)
            for w in words:
                for vb in ("remove","delete","forget","cancel","erase","drop","discard","trash","wipe","clear","purge","cancel"):
                    if _ed1(w, vb) <=1:
                        # extract payload after typo word
                        mv = re.search(rf"\b{re.escape(w)}\b\s*(?:the\s+|my\s+)?(?:task|reminder|fact|todo|item)?\s*(?:called\s+|named\s+)?\s*(.+)", low, re.IGNORECASE)
                        if mv and mv.group(1).strip():
                            m = mv
                            break
                if m:
                    break
    if m and not _looks_like_question(t):
        target = m.group(1).strip()
        # Strip leading "about", "a/the/my/new task/reminder/fact" if still present (e.g., "never mind about buy milk", "remove the task buy eggs")
        target = re.sub(r"^about\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"^(?:a\s+|the\s+|my\s+)?(?:new\s+)?(?:task|reminder|fact|todo|to-?do|item)\s+(?:called\s+|named\s+)?", "", target, flags=re.IGNORECASE).strip()
        if not target:
            return "What should I remove? Try 'remove buy milk'."
        # Try by ID first
        ids = [int(x) for x in re.findall(r"(?:\bid\s*)?(\d+)", target, re.IGNORECASE) if x.isdigit()]
        if ids and (re.search(r"\b(id|number|no\.?|#|item|one|two|three|four|five|six|seven|eight|nine)\b", target, re.IGNORECASE)
                    or re.fullmatch(r"[\s\d,andor]+", target, re.IGNORECASE)):
            done, missing = [], []
            for n in ids:
                row = _resolve_row(n)
                if row is None:
                    missing.append(n)
                elif _delete_row(row):
                    done.append(n)
                else:
                    missing.append(n)
            parts = []
            if done:
                parts.append("Okay, removed those." if len(done) > 1 else "Okay, removed that.")
            if missing:
                parts.append(f"Couldn't find number {', '.join(map(str, missing))}.")
            return " ".join(parts)
        # By title
        rows = _everything()
        match = None
        for r in rows:
            if (r["title"] or "").lower().startswith(target):
                match = r
                break
        if match is None:
            for r in rows:
                if target in (r["title"] or "").lower():
                    match = r
                    break
        if match is None:
            return f"Hmm, I can't find anything called '{target}' to remove."
        _delete_row(match)
        return "Okay, removed that."

    # ---- "Update fact N to <text>" (must run BEFORE generic update) ----
    m_fact = re.match(r"(?:update|change)\s+fact\s+(\d+)\s+(?:to\s+)?(.+)", low, re.IGNORECASE)
    if m_fact and not _looks_like_question(t):
        fid = int(m_fact.group(1))
        new_text = m_fact.group(2).strip()
        res = update_fact(fid, new_text)
        if res is None:
            return f"Hmm, I can't find fact {fid} — or that text already exists."
        return f"Got it — updated fact {fid} to '{new_text}'."

    # ---- "Update / Change / Reschedule / Move" ----
    # Patterns: "update X", "change X", "reschedule X", "move X to Y", "push X to Y"
    m = re.match(
        r"^(?:"
        r"update\s+(?:that\s+)?"
        r"|change\s+(?:that\s+)?"
        r"|reschedule\s+"
        r"|move\s+"
        r"|push\s+"
        r")(.+)",
        low, re.IGNORECASE)
    if m and not _looks_like_question(t):
        target = m.group(1).strip()
        # Check for "to <time>" or "for <time>"
        new_when = None
        mt = re.search(r"\b(?:to|for)\s+(.+)$", target, re.IGNORECASE)
        if mt:
            new_when_str = mt.group(1).strip()
            target = target[:mt.start()].strip()
            when, _ = _parse_when(new_when_str)
            new_when = when
        
        # Try by ID
        n = None
        if re.fullmatch(r"\d+", target):
            n = int(target)
        if n is not None:
            row = _resolve_row(n)
            if row is None:
                return f"Hmm, I can't find number {n} to update."
        else:
            rows = _everything()
            row = next((r for r in rows
                        if r["title"].lower().startswith(target) or target in (r["title"] or "").lower()),
                       None)
            if row is None:
                return f"Hmm, I can't find anything called '{target}' to update."
        if row["kind"] == "fact":
            return "That one's a fact I keep, not a to-do — say 'forget' plus its number if you want it gone."
        
        updates = {}
        if new_when:
            updates["when_at"] = new_when
        # Toggle completion if no time given
        if not new_when:
            updates["completed"] = not row["completed"]
        
        (update_reminder if row["kind"] == "reminder" else update_task)(row["store_id"], **updates)
        if new_when:
            return f"Done — '{row['title']}' moved to {friendly_when(new_when)}."
        state = "all done" if updates.get("completed") else "back on your list"
        return f"Done — '{row['title']}' is {state}."

    # ---- Facts: "I like X", "I love X", "I hate X", "my X is Y", "I'm allergic to X" ----
    # These are statements about the user, not commands
    m = re.match(
        r"^(?:"
        r"i\s+(?:like|love|enjoy|prefer|fancy)\s+"
        r"|i\s+(?:hate|dislike|can't stand|detest)\s+"
        r"|my\s+(?:favorite|favourite)\s+\w+\s+is\s+"
        r"|my\s+\w+\s+is\s+"
        r"|i'?m\s+(?:allergic\s+to|intolerant\s+to)\s+"
        r"|i\s+am\s+(?:allergic\s+to|intolerant\s+to)\s+"
        r"|remember\s+(?:that\s+)?i\s+(?:like|love|hate)\s+"
        r")(.+)",
        low, re.IGNORECASE)
    if m and not _looks_like_question(t):
        fact_text = m.group(1).strip()
        save_fact(fact_text)
        return f"Got it, I'll remember that."

    # ---- "What do you remember?" / "What are my tasks?" (tasks+reminders only) ----
    if re.match(
        r"^(what do you (remember|know)\??"
        r"|list my memor(ies|y)"
        r"|show my memor(ies|y)"
        r"|what have you (remembered|saved)\??"
        r"|what(?:'s| is) on (my )?(?:list|agenda|schedule|tasks?|reminders?)\??"
        r"|what are my (tasks?|reminders?)\??)$",
        low.strip(),
    ):
        rows = [r for r in _everything() if r["kind"] in ("task", "reminder")]
        if not rows:
            return "No tasks or reminders yet — just say 'remember ...' and I'll keep it."
        lines = ["Here are your tasks and reminders:"]
        lines += [_line(n, r) for n, r in enumerate(rows, 1)]
        lines.append("(tell me 'forget' plus the number to drop one)")
        return "\n".join(lines)

    # ---- "What are my facts?" / "What do you know about me?" (facts only) ----
    if re.match(
        r"^(what (are|'s| is) my facts\??"
        r"|what do you know about me\??"
        r"|tell me my facts\??"
        r"|list my facts\??"
        r"|show my facts\??)$",
        low.strip(),
    ):
        facts = [r for r in _everything() if r["kind"] == "fact"]
        if not facts:
            return "No facts saved yet — just say 'remember ...' and I'll keep it."
        lines = ["Here's what I know about you:"]
        lines += [_line(n, r) for n, r in enumerate(facts, 1)]
        lines.append("(tell me 'forget' plus the number to drop one)")
        return "\n".join(lines)

    # ---- "What are my facts?" / "What do you know about me?" (facts only) ----
    if re.match(
        r"^(what (are|'s| is) my facts\??"
        r"|what do you know about me\??"
        r"|tell me my facts\??"
        r"|list my facts\??"
        r"|show my facts\??)$",
        low.strip(),
    ):
        facts = [r for r in _everything() if r["kind"] == "fact"]
        if not facts:
            return "No facts saved yet — just say 'remember ...' or tell me something about yourself."
        lines = ["Here's what I know about you:"]
        lines += [_line(n, r) for n, r in enumerate(facts, 1)]
        lines.append("(tell me 'forget' plus the number to drop one)")
        return "\n".join(lines)

    # ---- Task/reminder questions: casual list or the one thing asked about ----
    if _is_task_question(low):
        return _answer_task_question(t)

    # ---- "Forget / Delete ..." ----
    m = re.match(r"forget\s+(?:that\s+)?(.+)", low)
    if not m:
        m = re.match(r"delete\s+(?:that\s+)?(.+)", low)
    if m:
        target = m.group(1).strip()
        ids = [int(x) for x in re.findall(r"(?:\bid\s*)?(\d+)", target, re.IGNORECASE)
               if x.isdigit()] if re.search(r"\d", target) else []
        id_like = bool(ids) and (
            bool(re.search(r"\b(id|number|no\.?|#|item|one|two|three|four|five|six|seven|eight|nine)\b",
                           target, re.IGNORECASE))
            or bool(re.fullmatch(r"[\s\d,andor]+", target, re.IGNORECASE)))
        if id_like:
            done, missing = [], []
            for n in ids:
                row = _resolve_row(n)
                if row is None:
                    missing.append(n)
                elif _delete_row(row):
                    done.append(n)
                else:
                    missing.append(n)
            parts = []
            if done:
                parts.append("Okay, I forgot those." if len(done) > 1 else "Okay, I forgot that.")
            if missing:
                parts.append(f"Hmm, I can't find number {', '.join(map(str, missing))}.")
            return " ".join(parts)
        # by title: exact/prefix first, then substring — across all three stores
        rows = _everything()
        match = None
        for r in rows:
            if (r["title"] or "").lower().startswith(target):
                match = r
                break
        if match is None:
            for r in rows:
                if target in (r["title"] or "").lower():
                    match = r
                    break
        if match is None:
            return f"Hmm, I can't find anything called '{target}' to forget."
        _delete_row(match)
        return "Okay, I forgot that."

    m = re.match(r"update\s+(?:that\s+)?(.+)", low)
    if not m:
        m = re.match(r"change\s+(?:that\s+)?(.+)", low)
    if m:
        target = m.group(1).strip()
        n = None
        if re.fullmatch(r"\d+", target):
            n = int(target)
        if n is not None:
            row = _resolve_row(n)
            if row is None:
                return f"Hmm, I can't find number {n} to update."
        else:
            rows = _everything()
            row = next((r for r in rows
                        if r["title"].lower().startswith(target) or target in (r["title"] or "").lower()),
                       None)
            if row is None:
                return f"Hmm, I can't find anything called '{target}' to update."
        if row["kind"] == "fact":
            return "That one's a fact — say 'update fact {n} to <new text>' to change it, or 'forget' plus its number."
        new_state = not row["completed"]
        (update_reminder if row["kind"] == "reminder" else update_task)(row["store_id"], completed=new_state)
        state = "all done" if new_state else "back on your list"
        return f"Done — '{row['title']}' is {state}."

    # ---- "Search / Find ..." ----
    m = re.match(r"^(?:search|find)\s+(?:for\s+)?(.+)", low, re.IGNORECASE)
    if m and not _looks_like_question(t):
        query = m.group(1).strip().strip("'\"")
        # strip optional type prefix like "tasks containing"
        query = re.sub(r"^(?:my\s+)?(?:tasks?|reminders?|facts?|items?)\s+(?:containing\s+|for\s+|with\s+)?", "", query, flags=re.IGNORECASE).strip()
        if not query:
            return "What should I search for? Try 'search for milk'."
        rows = _everything()
        hits = [r for r in rows if query.lower() in (r["title"] or "").lower()]
        if not hits:
            return f"Nothing found for '{query}'."
        lines = [f"Found {len(hits)} for '{query}':"]
        for h in hits[:8]:
            idx = rows.index(h) + 1
            lines.append(_line(idx, h))
        return "\n".join(lines)

    return None


def _is_task_question(low: str) -> bool:
    t = low.strip()
    task_words = ("task", "todo", "to-do", "reminder", "remind", "schedule",
                  "agenda", "due", "plan", "plans", "list", "upcoming", "coming up",
                  "appointment", "event")
    if any(w in t for w in task_words):
        return True
    return bool(re.match(
        r"^(do i have|have i got|is there|are there|am i free|am i busy|"
        r"when is my|when'?s my|what'?s (on|due|at)|what is (on|due|at)|"
        r"what do i have|what have i got|whats on|what's next|"
        r"what'?s happening|what is happening|what'?s going on|what is going on|"
        r"got anything|got any|any plans|anything (?:on|for|at|coming up)|"
        r"am i doing anything|do i have anything|have i got anything|"
        r"what time|when is|when'?s|what day)", t))


def _answer_task_question(orig: str) -> str | None:
    """Casual list answer, or just the one matching item. None = no opinion."""
    low = orig.lower().strip()
    rows = [r for r in _everything()
            if not r["completed"] and r["kind"] in ("task", "reminder")]
    free_check = "free" in low
    qwords = set(_words(orig))

    def _answer(matched, fallback_none: str) -> str:
        """Prefer the specific matching item; otherwise answer from `matched`."""
        if not matched:
            return fallback_none
        scored = sorted(
            ((len(qwords & set(_words(r["title"]))), r) for r in matched),
            key=lambda p: -p[0],
        )
        best, best_rows = scored[0][0], [r for s, r in scored if s == scored[0][0] and s > 0]
        if best > 0 and len(best_rows) == 1:
            r = best_rows[0]
            if re.match(r"\s*(when|what time)\b", low):
                return f"Your {_display(r)}."
            if free_check:
                return f"Nope — you've got {_display(r)}."
            return f"Yep — {_display(r)}."
        if best > 0:
            return "Those ones: " + _join_list([_display(r) for r in best_rows[:4]]) + "."
        return fallback_none

    # Parse specific time from query (e.g., "at 3pm", "at 6:30", "for 2pm tomorrow")
    target_time = None
    mt = re.search(r"\b(?:at|for|around|before|after)\s+(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", low, re.IGNORECASE)
    if mt:
        hour = int(mt.group(1))
        minute = int(mt.group(2) or 0)
        mer = (mt.group(3) or "").lower().replace(".", "")
        if "pm" in mer:
            hour = hour % 12 + 12
        elif "am" in mer:
            hour = hour % 12
        else:
            # No am/pm: assume pm for 1-6, am for 7-11, 12=noon
            if 1 <= hour <= 6:
                hour += 12
            elif hour == 12:
                hour = 12
        target_time = (hour, minute)

    # Which day are they asking about (if any)?
    target = None
    for w, off in (("tomorrow", 1), ("today", 0)):
        if re.search(rf"\b{w}\b", low):
            target = (datetime.now() + timedelta(days=off)).date()
    if target is None:
        for name, idx in _WEEKDAYS.items():
            if re.search(rf"\b{name}\b", low):
                delta = (idx - datetime.now().weekday()) % 7
                target = (datetime.now() + timedelta(days=delta)).date()

    if target is not None:
        day_rows = [r for r in rows if _row_on_date(r, target)]
        # Further filter by time if specified
        if target_time:
            hour, minute = target_time
            time_rows = [r for r in day_rows if _row_at_time(r, hour, minute)]
            if time_rows:
                # Return all items at this time (not just keyword matches)
                if free_check:
                    return f"Nope — you've got " + _join_list([_display(r) for r in time_rows[:4]]) + "."
                if len(time_rows) == 1:
                    return f"Yep — {_display(time_rows[0])}."
                return "At that time: " + _join_list([_display(r) for r in time_rows[:4]]) + "."
            if free_check:
                return f"Yep, you're free at {hour}:{minute:02d} — nothing at that time."
            return f"Nothing at {hour}:{minute:02d} on {target.strftime('%A, %B %d')}."
        
        specific = _answer(day_rows, fallback_none=None)
        if specific:
            return specific
        if not day_rows:
            if free_check:
                return "Yep, you're free — nothing like that on your list."
            others = [r for r in rows if not _row_on_date(r, target)]
            if others:
                today = datetime.now().date()
                tmrw = today + timedelta(days=1)
                dayword = ("today" if target == today
                           else "tomorrow" if target == tmrw
                           else target.strftime("%A, %B %d"))
                return (f"Nothing on {dayword} — you've got "
                        + _join_list([_display(r) for r in others[:4]]) + ".")
            return "Nothing like that on your list."
        if free_check:
            return "Nope — you've got " + _join_list([_display(r) for r in day_rows[:4]]) + "."
        return "On your list: " + _join_list([_display(r) for r in day_rows[:4]]) + "."

    if not rows:
        if free_check:
            return "Yep, you're free — nothing on your list."
        return "Nothing on your list — you're all clear."

    specific = _answer(rows, fallback_none=None)
    if specific:
        return specific

    if free_check:
        return "Yep, you're free — nothing like that on your list."

    if re.match(r"\s*(do i have|have i got|is there|are there)\b", low):
        topic = re.sub(r"^\s*(do i have|have i got|is there|are there)\s*",
                       "", low).strip(" ?")
        topic = re.sub(r"^(anything|something|any|a|an|the)\s+", "", topic).strip()
        if topic:
            return f"Nope, nothing about {topic} on your list."

    return "You've got a few things: " + _join_list([_display(r) for r in rows[:6]]) + "."


def get_memory_context() -> str:
    """Plain-language reminders/tasks/facts list for inclusion in prompts.

    The buddy checks this FIRST whenever the user asks about plans, reminders,
    or anything about themselves. Numbered the same way the recall reply is,
    so 'forget N' always lines up.
    """
    rows = _everything()
    if not rows:
        return "Reminders, tasks, and facts: nothing saved yet."
    lines = [
        "Your saved stuff (check this FIRST when the user asks about plans,",
        "reminders, or anything about themselves):",
    ]
    lines += [_line(n, r) for n, r in enumerate(rows, 1)]
    return "\n".join(lines)


def reset_all():
    """Clear all items and facts (--reset-memory)."""
    clear_tasks_reminders()
    clear_facts()


# Run one-time migration from old schema on import
migrate_from_old_schema()