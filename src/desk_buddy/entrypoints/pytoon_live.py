"""Live pygame player: type a line, watch a cartoon lip-sync it instantly.

Interactive mode:
    python -m desk_buddy.pytoon_live

Chat mode (Ollama replies, avatar speaks the reply):
    python -m desk_buddy.pytoon_live --chat
    python -m desk_buddy.pytoon_live --chat --ollama-model llama3.1:8b  # or any installed model

One-shot mode:
    python -m desk_buddy.pytoon_live "Hello, welcome to my application!"

Per line it: synthesizes speech (macOS 'say'), force-aligns phonemes with
Wav2Vec2, builds the cartoon mouth/pose frames, then plays frames in a window
in real time while the audio plays through the speakers. Closing the window
(or pressing Ctrl-C) stops playback. The window stays open forever until you
type quit/exit or close it.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime

# Package bootstrap: allow running this file directly (python .../pytoon_live.py)
if __package__ in (None, ""):
    __package__ = "desk_buddy.entrypoints"
    _SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, _SRC)
    import importlib as _ilib
    _ilib.import_module("desk_buddy")

from ..core.memory_handler import handle_user_text, get_memory_context
from ..core.agents import classify_intent, task_agent, reminder_agent, chat_agent
from ..prompts import SYSTEM_PROMPT, MEMORY_FACTS_BLOCK, MEMORY_SUMMARY_BLOCK
try:
    from ..core.crud_tools import TOOLS as CRUD_TOOLS, TOOL_FUNCS as CRUD_TOOL_FUNCS, get_tool_descriptions
except Exception:
    CRUD_TOOLS, CRUD_TOOL_FUNCS, get_tool_descriptions = [], {}, lambda: ""

from .._paths import ROOT as HERE, TMP_DIR

_ESPEAK_VOICE = None  # cached best espeak voice (probed once)


def _espeak_voice() -> str:
    """Pick the clearest available espeak voice (cached).

    The plain `en` voice is the most robotic. The +f3 female variants are
    markedly clearer and more natural. Variants are NOT listed by
    `espeak-ng --voices`, so probe with a trial synthesis instead of grep.
    """
    global _ESPEAK_VOICE
    if _ESPEAK_VOICE is not None:
        return _ESPEAK_VOICE
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    for candidate in ("en-us+f3", "en+f3", "en-us", "en"):
        if not espeak:
            break
        try:
            r = subprocess.run([espeak, "-v", candidate, "-w", os.devnull, "hi"],
                               capture_output=True, timeout=15)
            if r.returncode == 0:
                _ESPEAK_VOICE = candidate
                break
        except Exception:
            continue
    if _ESPEAK_VOICE is None:
        _ESPEAK_VOICE = "en"
    return _ESPEAK_VOICE


def _fix_env() -> None:
    """Auto-set certs + ffmpeg lib path so users don't need exports."""
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass
    fb = "/opt/homebrew/opt/ffmpeg/lib"
    if os.path.isdir(fb):
        cur = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if fb not in cur.split(":"):
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = fb + (":" + cur if cur else "")


def _ensure_venv() -> None:
    """Re-exec with the project venv python if pytoon is not importable."""
    try:
        import pytoon  # noqa: F401
        return
    except ImportError:
        pass

    venv_python = os.path.join(HERE, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        print(f"pytoon not available here; re-running with {venv_python}")
        os.execv(venv_python, [venv_python] + sys.argv)


_fix_env()
_ensure_venv()


def synthesize_speech(text: str, voice: str | None = None, rate: int = 180,
                      pitch: float = 1.0, chunk_id: str | None = None,
                      edge_voice: str | None = None) -> str:
    """Synthesizes text to an mp3 and returns its path.

    Tries the Edge neural voice first (Google-Assistant-like, needs network),
    then falls back to local espeak-ng.
    pitch > 1.0 raises the voice (chibi effect) without changing speed.
    chunk_id gives each sentence its own file so chunked playback doesn't clobber.
    On Linux (Pi) uses espeak-ng instantly if available, else macOS 'say'.
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    suffix = f"_{chunk_id}" if chunk_id is not None else ""
    script_file = os.path.join(TMP_DIR, f"live_script{suffix}.txt")
    aiff_file = os.path.join(TMP_DIR, f"live_speech{suffix}.aiff")
    wav_file = os.path.join(TMP_DIR, f"live_speech{suffix}.wav")
    mp3_file = os.path.join(TMP_DIR, f"live_speech{suffix}.mp3")

    # natural neural voice first (needs network); falls back below on failure
    try:
        from .pytoon_cli import edge_tts_synthesize

        edge_tts_synthesize(text, mp3_file, rate=rate, pitch=pitch, voice=edge_voice)
        return mp3_file
    except Exception as e:
        print(f"(edge TTS failed: {e}, falling back to local voice)")

    # Natural-voice path: espeak-ng with a clear +f3 variant, slow rate,
    # and a small word gap for clarity. (Plain `-v en` is the robotic one.)
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak and sys.platform != "darwin":
        # espeak rate is wpm, pitch 0-99; map our 1.0-1.4 pitch to espeak -p 50 + delta
        # for instant mode skip pitch entirely
        fast_skip_pitch = getattr(__import__('builtins'), '_PYTOON_FAST_SKIP_PITCH', False)
        try:
            cmd = [espeak, "-v", _espeak_voice(), "-s", str(rate), "-g", "5", "-w", wav_file, text]
            if not fast_skip_pitch and pitch and abs(pitch - 1.0) > 1e-6:
                # espeak -p 50 is default; scale to 60 for chibi
                p_val = int(50 * pitch)
                p_val = max(0, min(99, p_val))
                cmd = [espeak, "-v", _espeak_voice(), "-s", str(rate), "-g", "5",
                       "-p", str(p_val), "-w", wav_file, text]
            subprocess.run(cmd, check=True)
            # wav -> mp3 (no pitch filter needed, already pitched)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_file,
                           "-ac", "1", "-ar", "22050", "-codec:a", "libmp3lame", "-q:a", "6", mp3_file], check=True)
            return mp3_file
        except Exception:
            pass  # fall through to say

    with open(script_file, "w") as f:
        f.write(text)

    cmd = ["say", "-f", script_file, "-o", aiff_file, "-r", str(rate)]
    if voice:
        cmd += ["-v", voice]
    # if 'say' missing on Pi, try espeak again with aiff fallback
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        if espeak:
            subprocess.run([espeak, "-v", _espeak_voice(), "-s", str(rate), "-g", "5",
                            "-w", wav_file, text], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_file,
                           "-ac", "1", "-ar", "22050", "-codec:a", "libmp3lame", "-q:a", "6", mp3_file], check=True)
            return mp3_file
        raise

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", aiff_file]
    if pitch and abs(pitch - 1.0) > 1e-6:
        cmd += ["-filter:a", f"asetrate=22050*{pitch},aresample=22050,atempo={1.0 / pitch}"]
        print(f"(chibi pitch x{pitch})")
    cmd += ["-ac", "1", "-ar", "22050", "-codec:a", "libmp3lame", "-q:a", "6", mp3_file]
    subprocess.run(cmd, check=True)
    return mp3_file


def play_audio(path: str):
    """Starts non-blocking audio playback; returns the Popen handle."""
    if sys.platform == "darwin":
        return subprocess.Popen(["afplay", path])
    ffplay = shutil.which("ffplay")
    if ffplay:
        return subprocess.Popen([ffplay, "-nodisp", "-autoexit", "-loglevel", "error", path])
    raise RuntimeError("No audio player found (needs afplay or ffplay)")


def play_frames(frames, audio_path: str, fps: int = 30, max_width: int = 560,
                height_scale: float = 1.0) -> None:
    """Legacy one-shot player (kept for compatibility). Opens, plays, closes."""
    import numpy as np
    import pygame

    pygame.init()
    try:
        f0 = frames[0]
        disp_w = min(max_width, f0.shape[1])
        disp_h = int(f0.shape[0] * (disp_w / f0.shape[1]) * height_scale)
        screen = pygame.display.set_mode((disp_w, disp_h), pygame.RESIZABLE)
        pygame.display.set_caption("PyToon - live lip-sync")
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])

        proc = play_audio(audio_path)

        n = len(frames)
        t0 = time.monotonic()
        pacing = time.monotonic()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            elapsed = time.monotonic() - t0
            idx = int(elapsed * fps)
            if proc.poll() is not None and idx >= n:
                running = False

            idx = min(max(idx, 0), n - 1)
            rgb = np.ascontiguousarray(frames[idx][:, :, :3])
            surf = pygame.image.frombuffer(rgb, (rgb.shape[1], rgb.shape[0]), "RGB")
            w, h = screen.get_size()
            screen.blit(pygame.transform.smoothscale(surf, (w, h)), (0, 0))
            pygame.display.flip()

            while time.monotonic() - pacing < 1.0 / fps:
                time.sleep(0.002)
            pacing = time.monotonic()

        proc.terminate()
    finally:
        pygame.quit()


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter for chunked TTS — keeps latency low."""
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text.strip()) if p.strip()]
    # merge very short fragments (e.g. "Hi.") with next sentence
    merged: list[str] = []
    for p in parts:
        if merged and len(merged[-1].split()) <= 2 and len(p.split()) > 2:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged if len(" ".join(merged).split()) > 6 else [text.strip()]


def _extract_delta(chunk) -> str:
    """Handles both dict and object ollama stream chunks."""
    try:
        if isinstance(chunk, dict):
            return chunk.get("message", {}).get("content", "") or ""
        return getattr(getattr(chunk, "message", None), "content", "") or ""
    except Exception:
        return ""

def _think_for(model: str):
    """Disable hidden reasoning for realtime chat replies.

    Thinking models (qwen3, gemma3/4, deepseek-r1, ...) spend the tiny
    num_predict budget on hidden <think> tokens, leaving the visible reply
    empty ("(silence)"). Realtime avatar replies want instant visible text,
    so always request think=False.
    """
    return False


_THINK_TAG_RE = None  # lazy compiled
_META_PHRASES = ("user asked", "user said", "i need to respond", "i should ",
                 "i must ", "my response", "reasoning:", "let me think",
                 "common greeting", "greeting question", "plain text",
                 "no markdown", "no emojis", "stage direction",
                 "direct answer", "casual way", "1-2 sentences",
                 "1 - 2 sentences", "one or two sentences", "check in or start",
                 "intent classification", "which branch", "task branch",
                 "reminder branch", "chat branch", "crud tool", "intent=")


def _think_tag_re():
    global _THINK_TAG_RE
    if _THINK_TAG_RE is None:
        import re as _re
        _THINK_TAG_RE = _re.compile(r"<(?:think|thinking)>.*?(</(?:think|thinking)>|$)", _re.DOTALL | _re.IGNORECASE)
    return _THINK_TAG_RE


def _get_ollama_tools():
    """Convert CRUD_TOOLS to Ollama tool spec format."""
    if not CRUD_TOOLS:
        return None
    ollama_tools = []
    for t in CRUD_TOOLS:
        props = {}
        for k, v in t.get("parameters", {}).items():
            # infer type from description prefix
            typ = "string"
            if "int" in v.lower(): typ = "integer"
            elif "bool" in v.lower(): typ = "boolean"
            props[k] = {"type": typ, "description": v}
        required = [k for k, v in props.items() if "required" in v.get("description","").lower()]
        # task_create title is required, etc. fallback: first param required
        if not required and props:
            first = list(props.keys())[0]
            if t["name"] in ("task_create","reminder_create","fact_create","fact_update","item_search"):
                required = [first]
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {"type": "object", "properties": props, "required": required}
            }
        })
    return ollama_tools


def _execute_tool_call(name: str, args: dict):
    """Run a CRUD tool by name with args dict, return JSON-serializable result."""
    import json
    func = CRUD_TOOL_FUNCS.get(name)
    if not func:
        return {"error": f"Unknown tool {name}"}
    try:
        res = func(**(args or {}))
        # sanitize for JSON
        if isinstance(res, (list, dict, str, int, float, bool)) or res is None:
            # truncate huge lists for prompt
            if isinstance(res, list) and len(res) > 20:
                res = res[:20] + [{"truncated": f"{len(res)-20} more"}]
            return res
        return {"result": str(res)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _try_ollama_with_tools(model: str, messages: list, opts: dict, max_iters: int = 2):
    """Try Ollama tool calling loop; fallback to normal chat if tools unsupported."""
    tools = _get_ollama_tools()
    if not tools:
        import ollama
        return ollama.chat(model=model, messages=messages, options=opts, think=_think_for(model))
    import ollama, json
    cur_messages = list(messages)
    for _ in range(max_iters):
        try:
            resp = ollama.chat(model=model, messages=cur_messages, options=opts, tools=tools, think=False)
        except TypeError:
            # older ollama client without tools support
            import ollama as _o
            return _o.chat(model=model, messages=messages, options=opts, think=_think_for(model))
        except Exception:
            raise
        tcalls = getattr(getattr(resp, "message", resp), "tool_calls", None)
        # some versions put tool_calls on dict
        if isinstance(resp, dict):
            tcalls = resp.get("message", {}).get("tool_calls")
        if not tcalls:
            return resp
        # execute each tool call and append results
        try:
            content = getattr(resp.message, "content", "") or ""
        except Exception:
            content = ""
        cur_messages.append({"role": "assistant", "content": content, "tool_calls": [
            {"function": {"name": c.function.name, "arguments": c.function.arguments}} if hasattr(c, "function") else c for c in tcalls
        ]})
        for call in tcalls:
            if hasattr(call, "function"):
                name = call.function.name
                args = call.function.arguments or {}
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except: args = {}
            else:
                name = call.get("function", {}).get("name")
                args = call.get("function", {}).get("arguments", {})
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except: args = {}
            result = _execute_tool_call(name, args)
            cur_messages.append({"role": "tool", "content": json.dumps(result, default=str)})
        # loop to let LLM synthesize final answer from tool results
    # after max_iters, final synthesis
    import ollama
    return ollama.chat(model=model, messages=cur_messages, options=opts, think=False)


def _is_meta_sentence(s: str) -> bool:
    low = (s or "").strip().lower()
    if not low:
        return True
    if low.startswith("hmm,") and ("user" in low or "greeting" in low):
        return True
    return any(p in low for p in _META_PHRASES)


def clean_reply(text: str) -> str:
    """Strip thinking leakage so the avatar only says the final answer.

    Removes <think>...</think> blocks (including unclosed) and drops
    meta-narration sentences ("Hmm, the user asked...", "I need to...").
    Never returns meta text; may return "" if nothing answer-like remains.
    """
    import re
    t = _think_tag_re().sub("", text or "").strip()
    if not t:
        return ""
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', t) if p.strip()]
    kept = [p for p in parts if not _is_meta_sentence(p)]
    return " ".join(kept).strip()


def _get_time_greeting() -> str:
    """Returns a time-appropriate greeting."""
    from datetime import datetime
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"


def _generate_welcome_message(args) -> str | None:
    """Generates a welcome message from the LLM based on current time.
    Only used in --chat mode."""
    if not getattr(args, "chat", False):
        return None
    try:
        import ollama
        greeting = _get_time_greeting()
        current_time = datetime.now().strftime("%I:%M %p")
        prompt = (
            f"You're Desk Buddy — a friend who lives on this desk. "
            f"It's {current_time}. "
            f"Start with exactly: '{greeting},' "
            "then add ONE short, casual follow-up. Like a real person greeting a friend. "
            "No markdown, no emojis, no customer service energy. "
            "Examples: 'Good morning, coffee's ready if you want some.' "
            "'Hey, long day?' 'Evening. How'd it go?'"
        )
        resp = ollama.chat(
            model=args.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 32, "num_ctx": 512, "temperature": 0.8,
                     "num_thread": 4, "keep_alive": "10m"},
            think=False
        )
        msg = resp.message.content.strip()
        # Clean any thinking tags
        import re
        msg = re.sub(r"יתוח.*?(ןוית|$)", "", msg, flags=re.DOTALL | re.IGNORECASE).strip()
        return msg
    except Exception:
        return None


def _refresh_sqlite_context(history: list) -> None:
    """Rebuild the reminders/tasks block in the system message from scratch.

    Always reflects the current DB contents (old block is stripped first so
    newly remembered items show up immediately instead of going stale).
    """
    ctx = get_memory_context()
    for m in history:
        if m.get("role") == "system":
            # strip any previously appended memory blocks (old markers)
            base = m["content"]
            for marker in [
                "\n\nCurrent memories from SQLite:\n",
                "\n\nReminders and tasks",
                "\n---\n\nYour memory (check this FIRST",
                "\n---\n\nYour long-term memory about them:",
            ]:
                if marker in base:
                    base = base.split(marker)[0]
            m["content"] = base + "\n\n---\n\nYour memory (check this FIRST when they ask about plans, reminders, or themselves):\n" + ctx
            break


def _append_user_once(history: list, user_text: str) -> None:
    """Append the user message unless it is already the last message.

    Prevents duplicate user turns when the streaming path falls back to
    chat_reply after a mid-stream failure.
    """
    if history and history[-1].get("role") == "user" and history[-1].get("content") == user_text:
        return
    history.append({"role": "user", "content": user_text})


def _preload_models(args, show_progress=True) -> None:
    """Preloads Ollama + Wav2Vec2 models synchronously before window opens.
    
    This eliminates the first-turn latency by warming up:
    - Ollama model (kept alive via keep_alive="10m")
    - Wav2Vec2 ASR model (torchaudio pipeline)
    - Torch thread configuration for Pi
    """
    if show_progress:
        print("Preloading AI models...", flush=True)

    # Configure torch threads early (helps on Pi)
    try:
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        if show_progress:
            print("  ✓ Torch threads configured", flush=True)
    except Exception:
        if show_progress:
            print("  ⚠ Torch not available", flush=True)

    # Preload Ollama model (chat mode only)
    if getattr(args, "chat", False):
        try:
            import ollama
            if show_progress:
                print(f"  Loading {args.ollama_model}...", flush=True)
            ollama.chat(model=args.ollama_model,
                        messages=[{"role": "user", "content": "hi"}],
                        options={"num_predict": 1, "num_ctx": 512, "keep_alive": "10m"})
            if show_progress:
                print(f"  ✓ Ollama ({args.ollama_model}) ready", flush=True)
        except Exception as e:
            if show_progress:
                print(f"  ⚠ Ollama preload failed: {e}", flush=True)

    # Preload Wav2Vec2 ASR model (used for phoneme alignment)
    try:
        if show_progress:
            print("  Loading Wav2Vec2...", flush=True)
        from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
        WAV2VEC2_ASR_BASE_960H.get_model()
        if show_progress:
            print("  ✓ Wav2Vec2 ready", flush=True)
    except Exception as e:
        if show_progress:
            print(f"  ⚠ Wav2Vec2 preload failed: {e}", flush=True)

    if show_progress:
        print("All models loaded — opening window...", flush=True)


def run_persistent(args, initial_text=None) -> None:
    """Rebuild the reminders/tasks block in the system message from scratch.

    Always reflects the current DB contents (old block is stripped first so
    newly remembered items show up immediately instead of going stale).
    """
    ctx = get_memory_context()
    for m in history:
        if m.get("role") == "system":
            # strip any previously appended block (old + new marker)
            base = m["content"].split("\n\nCurrent memories from SQLite:\n")[0]
            base = base.split("\n\nReminders and tasks")[0]
            m["content"] = base + "\n\n" + ctx
            break


def chat_reply(user_text: str, history: list, model: str, fast: bool = False) -> str:
    """Asks Ollama for a short spoken reply. Updates history in place."""
    import ollama

    # SQLite memory commands are handled locally, no LLM call needed
    mem_status = handle_user_text(user_text)
    if mem_status:
        _append_user_once(history, user_text)
        history.append({"role": "assistant", "content": mem_status})
        return mem_status

    # Fresh SQLite memory context so the LLM sees newly stored facts
    _refresh_sqlite_context(history)

    _append_user_once(history, user_text)
    # Pi CPU: keep prompt small — system + last 8 messages only.
    # 20 turns = 500+ tokens prefill = 60s+ stall then client timeout (500).
    short_history = ([history[0]] + history[-8:]) if history and history[0].get("role") == "system" else history[-8:]
    # Small budgets = fast on Pi CPU. 40 tokens ~ 2 sentences, 1536 ctx = 30% faster prefill.
    # num_thread 4 = Pi sweet spot; otherwise Ollama oversubscribes.
    # keep_alive keeps model resident across turns (no reload).
    opts = {"num_predict": 40, "num_ctx": 1536, "temperature": 0.7, "top_p": 0.9,
            "repeat_penalty": 1.05, "num_thread": 4, "keep_alive": "10m"}
    print(f"(asking {model}... ~5-15s on Pi CPU)", flush=True)
    # Try tool calling first (all CRUD available), fallback to plain chat
    raw = None
    try:
        try:
            resp = _try_ollama_with_tools(model, short_history, opts)
        except Exception as e:
            # Auto-pull if model missing
            msg = str(e)
            if "not found" in msg or "404" in msg:
                try:
                    import ollama as _oll
                    print(f"(pulling missing model {model}...)")
                    _oll.pull(model)
                    resp = _try_ollama_with_tools(model, short_history, opts)
                except Exception:
                    raise e
            else:
                raise e
        # normalize response extraction for both plain and tool paths
        try:
            raw = resp.message.content.strip() if hasattr(resp.message, "content") else resp["message"]["content"].strip()
        except Exception:
            raw = getattr(getattr(resp, "message", resp), "content", "") or ""
    except Exception:
        import ollama as _oll
        try:
            resp = _oll.chat(model=model, messages=short_history, options=opts, think=_think_for(model))
        except Exception as e:
            msg = str(e)
            if "not found" in msg or "404" in msg:
                try:
                    print(f"(pulling missing model {model}...)")
                    _oll.pull(model)
                    resp = _oll.chat(model=model, messages=short_history, options=opts, think=_think_for(model))
                except Exception:
                    raise e
            else:
                raise e
        raw = resp.message.content.strip()
    reply = clean_reply(raw or "")
    if not reply:
        # Thinking model spent its budget narrating; one retry demanding
        # only the final answer (still cheaper than speaking meta aloud).
        retry = short_history + [{"role": "user",
                                  "content": "Reply with ONLY the final answer, no thinking or narration."}]
        try:
            resp2 = ollama.chat(model=model, messages=retry, options=opts, think=False)
            reply = clean_reply(resp2.message.content.strip()) or resp2.message.content.strip()
        except Exception:
            reply = raw or ""
    history.append({"role": "assistant", "content": reply})
    return reply


def chat_reply_streaming(user_text: str, history: list, model: str, args, play_queue, intent) -> str:
    """Streams Ollama tokens and builds+queues each sentence instantly. Returns full reply."""
    import ollama

    # SQLite memory commands are handled locally, no LLM call needed
    mem_status = handle_user_text(user_text)
    if mem_status:
        _append_user_once(history, user_text)
        history.append({"role": "assistant", "content": mem_status})
        print(f"Avatar: {mem_status}")
        try:
            frames, audio, seconds = build_frames_for_line(mem_status, args, intent, chunk_id=uuid.uuid4().hex[:6])
            play_queue.put((frames, audio, seconds))
        except Exception:
            pass
        return mem_status

    # Fresh SQLite memory context so the LLM sees newly stored facts
    _refresh_sqlite_context(history)

    _append_user_once(history, user_text)
    short_history = ([history[0]] + history[-8:]) if history and history[0].get("role") == "system" else history[-8:]
    # Small prompt + small budget: only way llama finishes before timeout on Pi.
    # 1536 ctx + 40 predict = ~40% faster than 2048/48
    opts = {"num_predict": 40, "num_ctx": 1536, "temperature": 0.7, "top_p": 0.9,
            "repeat_penalty": 1.05, "num_thread": 4, "keep_alive": "10m"}
    # instant: skip chibi pitch filter unless user explicitly set --pitch (saves ffmpeg pass)
    import builtins
    if getattr(args, "pitch", None) is not None:
        builtins._PYTOON_FAST_SKIP_PITCH = False
    else:
        builtins._PYTOON_FAST_SKIP_PITCH = True

    try:
        stream = ollama.chat(model=model, messages=short_history, options=opts, stream=True, think=_think_for(model))
    except Exception as e:
        msg = str(e)
        if "not found" in msg or "404" in msg:
            try:
                print(f"(pulling missing model {model}...)")
                ollama.pull(model)
                stream = ollama.chat(model=model, messages=short_history, options=opts, stream=True, think=_think_for(model))
            except Exception:
                raise e
        else:
            raise e
    full = ""
    buf = ""
    first_sent = True
    sent_count = 0
    # sentence flush helper
    def flush_sentence(s: str):
        nonlocal first_sent, sent_count
        s = clean_reply(s.strip())
        if not s or _is_meta_sentence(s):
            return  # never speak thinking aloud
        print(f"Avatar: {s}" if sent_count == 0 else f"  + {s}")
        try:
            frames, audio, seconds = build_frames_for_line(s, args, intent, chunk_id=uuid.uuid4().hex[:6])
            play_queue.put((frames, audio, seconds))
            if first_sent:
                print(f"▶ first audio {seconds:.1f}s — playing instantly, rest streaming...")
                first_sent = False
            sent_count += 1
        except Exception as e:
            print(f"(stream build failed for {s[:30]!r}: {e})")

    for chunk in stream:
        delta = _extract_delta(chunk)
        if not delta:
            continue
        full += delta
        buf += delta
        # flush whenever buf ends with sentence punct and is worth speaking
        # we keep minimal buffer so first sentence ships ASAP
        stripped = buf.strip()
        if len(stripped.split()) >= 3 and stripped[-1] in ".!?" and len(stripped) > 10:
            # check if we have 1+ complete sentences; split and keep incomplete tail
            # for instant, flush the whole buf as one sentence immediately
            if re.search(r'[.!?]\s*$', stripped):
                flush_sentence(stripped)
                buf = ""

    # flush remainder (no trailing punct or short tail)
    if buf.strip():
        flush_sentence(buf.strip())

    cleaned = clean_reply(full)
    if not cleaned:
        # Only thinking/meta arrived (or nothing) — fall back to one
        # non-streaming call instead of saving "(silence)" or speaking meta,
        # which would poison history and queue wrong audio.
        try:
            delattr(builtins, '_PYTOON_FAST_SKIP_PITCH')
        except Exception:
            pass
        fallback = chat_reply(user_text, history, model, fast=True)
        # chat_reply updated history but queued no audio; speak it now.
        if fallback.strip() and fallback.strip() != "(silence)":
            print(f"Avatar: {fallback.strip()}")
            try:
                frames, audio, seconds = build_frames_for_line(
                    fallback.strip(), args, intent, chunk_id=uuid.uuid4().hex[:6])
                play_queue.put((frames, audio, seconds))
            except Exception as e:
                print(f"(fallback build failed: {e})")
        return fallback.strip()
    history.append({"role": "assistant", "content": cleaned})
    # clear fast flag
    try:
        delattr(builtins, '_PYTOON_FAST_SKIP_PITCH')
    except Exception:
        pass
    return cleaned


DEFAULT_HISTORY_FILE = os.path.join(TMP_DIR, "chat_history.json")
DEFAULT_FACTS_FILE = os.path.join(TMP_DIR, "user_facts.json")
DEFAULT_MEMORY_MD = os.path.join(TMP_DIR, "MEMORY.md")
COMPACT_KEEP_RECENT = 12  # turns kept verbatim when compacting


def load_history(path: str, system: str) -> list:
    """Loads persisted chat history; falls back to fresh [system]."""
    import json

    if path and os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            msgs = data.get("messages", []) if isinstance(data, dict) else data
            msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")]
            if msgs:
                if msgs[0].get("role") == "system":
                    msgs[0]["content"] = system  # refresh system prompt
                else:
                    msgs.insert(0, {"role": "system", "content": system})
                return msgs
        except Exception as e:
            print(f"(could not load memory {path}: {e}, starting fresh)")
    return [{"role": "system", "content": system}]


def save_history(path: str, history: list, keep: int = 40) -> None:
    """Persists system + last `keep` turns (non-blocking-safe, tiny file)."""
    if not path:
        return
    import json

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sys_msg = [history[0]] if history and history[0].get("role") == "system" else []
        tail = [m for m in history[1:] if m.get("role") in ("user", "assistant")][-keep:]
        with open(path, "w") as f:
            json.dump({"messages": sys_msg + tail}, f)
    except Exception as e:
        print(f"(could not save memory: {e})")


def load_facts(path: str) -> list:
    import json

    if path and os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            facts = data.get("facts", []) if isinstance(data, dict) else []
            return [str(x) for x in facts if str(x).strip()][:50]
        except Exception:
            pass
    return []


def save_facts(path: str, facts: list) -> None:
    if not path:
        return
    import json

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"facts": [str(x) for x in facts if str(x).strip()][:50]}, f)
    except Exception:
        pass


def refresh_facts_background(user_text: str, reply: str, facts_path: str, model: str) -> None:
    """Fire-and-forget: asks Ollama to update long-term user facts."""
    import threading

    def _work():
        try:
            import json
            import ollama

            facts = load_facts(facts_path)
            prompt = (
                "You maintain a list of stable facts about the user (name, likes, "
                "preferences, ongoing topics). Given the existing facts and this "
                "exchange, output the updated facts as a JSON array of short strings. "
                "Keep at most 20 facts, each under 15 words. Drop nothing still true. "
                "If nothing worth remembering, return the existing facts unchanged.\n\n"
                f"Existing facts: {json.dumps(facts)}\n"
                f"User said: {user_text}\nAssistant replied: {reply}\n"
                "Return ONLY the JSON array."
            )
            resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                                 options={"num_predict": 64, "num_ctx": 512, "temperature": 0.2,
                                          "keep_alive": "10m"},
                                 think=_think_for(model))
            text = resp.message.content.strip()
            start, end = text.find("["), text.rfind("]")
            if start != -1 and end != -1:
                new_facts = json.loads(text[start:end + 1])
                if isinstance(new_facts, list):
                    save_facts(facts_path, new_facts)
        except Exception:
            pass  # memory must never break the show

    threading.Thread(target=_work, daemon=True).start()


def load_memory_md(path: str) -> dict:
    """Reads MEMORY.md -> {'facts': [...], 'summary': '...'}. Missing file -> empty."""
    mem = {"facts": [], "summary": ""}
    if not path or not os.path.exists(path):
        return mem
    try:
        with open(path) as f:
            text = f.read()
        section = None
        summary_lines: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if line.lower().startswith("## facts"):
                section = "facts"
            elif line.lower().startswith("## conversation summary"):
                section = "summary"
            elif line.startswith("#"):
                section = None
            elif section == "facts" and line.startswith("-"):
                fact = line[1:].strip()
                if fact:
                    mem["facts"].append(fact)
            elif section == "summary" and raw.strip():
                summary_lines.append(raw.strip())
        mem["summary"] = "\n".join(summary_lines).strip()
    except Exception:
        pass
    return mem


def save_memory_md(path: str, facts: list, summary: str) -> None:
    """Writes the single human-readable memory file (facts + summary)."""
    if not path:
        return
    try:
        from datetime import datetime

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        clean_facts = [str(x).strip() for x in facts if str(x).strip()][:20]
        lines = [
            "# Avatar memory",
            f"Updated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Facts",
        ]
        lines += [f"- {f}" for f in clean_facts] or ["- (nothing remembered yet)"]
        lines += ["", "## Conversation summary", (summary or "(no older conversations summarized yet)").strip(), ""]
        with open(path, "w") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print(f"(could not save memory file: {e})")


def inject_memory(history: list, base_system: str, facts: list, summary: str) -> None:
    """Folds facts + summary into the system message so Ollama always sees them."""
    if not history:
        return
    from datetime import datetime
    # Replace the datetime placeholder in the base system prompt
    current_dt = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    system_with_dt = base_system.replace("{current_datetime}", current_dt)
    
    parts = [system_with_dt]
    if facts:
        parts.append(MEMORY_FACTS_BLOCK + "\n".join(f"- {f}" for f in facts))
    if summary:
        parts.append(MEMORY_SUMMARY_BLOCK + summary)
    if history[0].get("role") == "system":
        history[0]["content"] = "\n".join(parts)
    else:
        history.insert(0, {"role": "system", "content": "\n".join(parts)})


def compact_memory(history: list, args) -> bool:
    """If history grew past --compact-at turns, summarize the old half into
    MEMORY.md (keeping only important stuff) and trim history to recent turns.
    Returns True if a compaction happened."""
    import json

    compact_at = getattr(args, "compact_at", 40)
    turns = [m for m in history if m.get("role") in ("user", "assistant")]
    if len(turns) <= compact_at:
        return False
    keep_turns = [m for m in history[1:] if m.get("role") in ("user", "assistant")][-COMPACT_KEEP_RECENT:]
    drop_turns = [m for m in history[1:] if m.get("role") in ("user", "assistant")][:len(turns) - COMPACT_KEEP_RECENT]
    if not drop_turns:
        return False

    mem_path = getattr(args, "memory_md", None) or DEFAULT_MEMORY_MD
    old_mem = load_memory_md(mem_path)
    transcript = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in drop_turns)[:6000]
    try:
        import ollama

        prompt = (
            "You compact a chatbot's memory. Below are EXISTING facts, an EXISTING "
            "summary, and an OLD conversation that will be deleted. Merge everything "
            "into updated memory, keeping ONLY durable important info (user identity, "
            "preferences, ongoing topics, decisions). Drop greetings and chit-chat.\n"
            "Return ONLY JSON: {\"facts\": [max 20 short strings], "
            "\"summary\": \"2-8 sentences, max 1500 chars\"}.\n\n"
            f"Existing facts: {json.dumps(old_mem['facts'])}\n"
            f"Existing summary: {old_mem['summary'] or '(none)'}\n"
            f"Old conversation to absorb and delete:\n{transcript}"
        )
        resp = ollama.chat(model=args.ollama_model, messages=[{"role": "user", "content": prompt}],
                             options={"num_predict": 512, "num_ctx": 2048, "temperature": 0.2,
                                      "keep_alive": "10m"},
                             think=_think_for(args.ollama_model))
        text = resp.message.content.strip()
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1])
        facts = [str(x).strip() for x in data.get("facts", []) if str(x).strip()][:20]
        summary = str(data.get("summary", "")).strip()[:1500]
        save_memory_md(mem_path, facts, summary)
        save_facts(getattr(args, "facts_file", None) or DEFAULT_FACTS_FILE, facts)
        history[:] = [history[0]] + keep_turns
        inject_memory(history, args.system, facts, summary)
        save_history(getattr(args, "memory_file", None) or DEFAULT_HISTORY_FILE, history)
        print(f"(memory compacted: {len(drop_turns)} old turns summarized into {os.path.basename(mem_path)}, {len(keep_turns)} recent kept)")
        return True
    except Exception as e:
        print(f"(memory compaction skipped: {e})")
        return False


def build_frames_for_line(line: str, args, intent: str | None = None, chunk_id: str | None = None):
    """Synthesizes + aligns one line. Returns (frames, audio_path, seconds)."""
    from pytoon.animator import animate
    from .pytoon_cli import resolve_voice, resolve_pitch

    voice, rate, edge_voice = resolve_voice(line, args, intent)
    # in fast mode bump rate +10% and optionally skip chibi pitch filter
    if getattr(args, "fast", False):
        rate = int(rate * 1.1)
        pitch = 1.0 if getattr(args, "no_chibi_fast", False) else resolve_pitch(args)
    else:
        pitch = resolve_pitch(args)
    audio = synthesize_speech(line, voice=voice, rate=rate, pitch=pitch, chunk_id=chunk_id,
                              edge_voice=edge_voice)
    use_cube = getattr(args, "forward", False) or getattr(args, "cube", True)
    if use_cube:
        from ..avatar.front_face import viseme_frames_for, render_forward_frame
        from .pytoon_cli import detect_mood

        try:
            fw, fh = (int(v) for v in str(args.closeup_size).lower().split("x"))
        except ValueError:
            sys.exit(f"error: --closeup-size must be WxH, got {args.closeup_size!r}")
        # Face mirrors the reply's feeling (keyword scan; LLM only if --mood-llm).
        mood_model = getattr(args, "ollama_model", None) if getattr(args, "mood_llm", False) else None
        mood = detect_mood(line, mood_model)
        # print(f"(cube mood: {mood})")
        # print("Aligning phonemes and building cube-bot frames...")
        visemes = viseme_frames_for(audio, line, args.fps)
        frames = [render_forward_frame(v, i, args.fps, (fw, fh), mood) for i, v in enumerate(visemes)]
        seconds = len(frames) / args.fps
    else:
        # print("Aligning phonemes and building frames...")
        animation = animate(audio_file=audio, transcript=line, fps=args.fps)
        frames = animation.final_frames
        if args.closeup:
            from .pytoon_cli import closeup_frame

            try:
                cw, ch = (int(v) for v in str(args.closeup_size).lower().split("x"))
            except ValueError:
                sys.exit(f"error: --closeup-size must be WxH, got {args.closeup_size!r}")
            # print("Cropping to face close-up...")
            frames = [closeup_frame(animation, i, (cw, ch)) for i in range(len(frames))]
        seconds = len(animation.final_frames) / animation.fps
    return frames, audio, seconds


def run_persistent(args, initial_text=None) -> None:
    """Persistent window that stays open forever. Type anytime, it speaks.

    - pygame runs in the main thread (required on macOS)
    - stdin reader + speech/align builder run in background threads
    - window never closes until you close it or type quit/exit
    """
    import queue
    import threading
    import numpy as np
    import pygame

    build_queue: "queue.Queue[str | None]" = queue.Queue()
    play_queue: "queue.Queue[tuple | None]" = queue.Queue()
    stop = threading.Event()
    building = threading.Event()  # set while a reply/speech is being built
    interrupt = threading.Event()  # set when new input should interrupt playback

    # Turn counter: >0 means a turn is in progress (building or playing)
    # input_loop waits for this to reach 0 before accepting new input
    turn_counter = [0]  # Use list for mutable closure
    turn_lock = threading.Lock()

    def _turn_start():
        with turn_lock:
            turn_counter[0] += 1
            # print(f"[DEBUG] turn_start: {turn_counter[0]}")

    def _turn_end():
        with turn_lock:
            turn_counter[0] = max(0, turn_counter[0] - 1)
            # print(f"[DEBUG] turn_end: {turn_counter[0]}")

    def _turns_active() -> bool:
        with turn_lock:
            return turn_counter[0] > 0

    def _wait_for_turn_done():
        """Block until no turns are active."""
        while not stop.is_set():
            with turn_lock:
                if turn_counter[0] == 0:
                    return
            time.sleep(0.05)

    def input_loop():
        while not stop.is_set():
            # Wait until previous turn is fully done (built + played)
            _wait_for_turn_done()
            if stop.is_set():
                break
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                build_queue.put(None)
                return
            if line.lower() in {"quit", "exit", "q"}:
                build_queue.put(None)
                return
            if not line:
                continue
            # Mark that we're now processing a turn
            _turn_start()
            build_queue.put(line)

    def builder_loop():
        history = []
        facts_path = getattr(args, "facts_file", None) or DEFAULT_FACTS_FILE
        mem_path = getattr(args, "memory_md", None) or DEFAULT_MEMORY_MD
        if getattr(args, "chat", False):
            if getattr(args, "reset_memory", False):
                for p in (getattr(args, "memory_file", None) or DEFAULT_HISTORY_FILE, facts_path, mem_path):
                    try:
                        if p and os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                from ..core.memory_handler import reset_all
                try:
                    reset_all()
                except Exception as e:
                    print(f"(could not clear memory stores: {e})")
                print("(memory cleared)")
            history = load_history(getattr(args, "memory_file", None) or DEFAULT_HISTORY_FILE, args.system)
            mem = load_memory_md(mem_path)
            facts = mem["facts"] or load_facts(facts_path)  # legacy json fallback
            if mem["facts"] != facts and facts:
                save_memory_md(mem_path, facts, mem["summary"])  # import legacy facts
            past = len([m for m in history if m.get("role") in ("user", "assistant")])
            if past or facts or mem["summary"]:
                print(f"(remembering {past} past messages, {len(facts)} facts)")
                for fc in facts:
                    print(f"  - {fc}")
            inject_memory(history, args.system, facts, mem["summary"])
        while not stop.is_set():
            line = build_queue.get()
            if line is None:
                play_queue.put(None)
                return
            building.set()
            try:
                speak = line
                intent = None
                if getattr(args, "chat", False):
                    # === MULTI-AGENT: Agent 1 = Intent Classifier (logs to .tmp/intent.log) ===
                    mem_now = load_memory_md(mem_path)
                    facts_now = mem_now["facts"] or load_facts(facts_path)
                    if facts_now != (mem_now["facts"] or []) and facts_now:
                        save_memory_md(mem_path, facts_now, mem_now["summary"])
                        mem_now["facts"] = facts_now
                    inject_memory(history, args.system, facts_now, mem_now["summary"])
                    intent = classify_intent(line, args.ollama_model)
                    print(f"[Agent 1] intent={intent} -> routing")
                    # write thinking trace for visibility
                    think_path = os.path.join(TMP_DIR, "thinking.log")
                    try:
                        os.makedirs(TMP_DIR, exist_ok=True)
                        with open(think_path, "a") as tf:
                            tf.write(f"{datetime.now().isoformat()} | prompt={line!r} | intent={intent}\n")
                    except:
                        pass
                    if intent == "TASK":
                        print("[Agent 2: TaskAgent] handling...")
                        # Try fast local SQLite handler first (human, no LLM, handles list/search/create)
                        mem_reply = handle_user_text(line)
                        if mem_reply is not None:
                            speak = mem_reply
                            print(f"TaskAgent (local): {speak}")
                        else:
                            speak = task_agent(line, history, args.ollama_model)
                            print(f"TaskAgent: {speak}")
                        history.append({"role": "user", "content": line})
                        history.append({"role": "assistant", "content": speak})
                        frames, audio, seconds = build_frames_for_line(speak, args, intent, chunk_id=uuid.uuid4().hex[:6])
                        play_queue.put((frames, audio, seconds))
                    elif intent == "REMINDER":
                        print("[Agent 2: ReminderAgent] handling...")
                        mem_reply = handle_user_text(line)
                        if mem_reply is not None:
                            speak = mem_reply
                            print(f"ReminderAgent (local): {speak}")
                        else:
                            speak = reminder_agent(line, history, args.ollama_model)
                            print(f"ReminderAgent: {speak}")
                        history.append({"role": "user", "content": line})
                        history.append({"role": "assistant", "content": speak})
                        frames, audio, seconds = build_frames_for_line(speak, args, intent, chunk_id=uuid.uuid4().hex[:6])
                        play_queue.put((frames, audio, seconds))
                    else:  # CHAT
                        print("Thinking (Ollama streaming)...")
                        try:
                            speak = chat_reply_streaming(line, history, args.ollama_model, args, play_queue, intent)
                        except Exception as e:
                            import traceback
                            print(f"(stream failed: {type(e).__name__}: {e}, falling back)")
                            traceback.print_exc()
                            speak = chat_reply(line, history, args.ollama_model, fast=True)
                            print(f"Avatar: {speak}")
                            sentences = _split_sentences(speak) if len(speak) > 40 else [speak]
                            for sent in sentences:
                                frames, audio, seconds = build_frames_for_line(sent, args, intent, chunk_id=uuid.uuid4().hex[:6])
                                play_queue.put((frames, audio, seconds))
                    save_history(getattr(args, "memory_file", None) or DEFAULT_HISTORY_FILE, history)
                    if getattr(args, "auto_facts", False):
                        refresh_facts_background(line, speak, facts_path, args.ollama_model)
                    compact_memory(history, args)
                else:
                    # echo mode: speak your line directly
                    print("Synthesizing speech...")
                    frames, audio, seconds = build_frames_for_line(speak, args, intent, chunk_id=uuid.uuid4().hex[:6])
                    print(f"Queued {seconds:.1f}s — playing next...")
                    play_queue.put((frames, audio, seconds))
            except Exception as e:  # keep window alive on errors
                import traceback
                print(f"error building animation: {type(e).__name__}: {e}")
                traceback.print_exc()
                _turn_end()  # Allow retry on error
            finally:
                building.clear()

    # Preload models BEFORE opening window (blocking, with progress)
    _preload_models(args, show_progress=True)

    # Generate welcome message from LLM (chat mode only)
    welcome_msg = None
    if getattr(args, "chat", False):
        print("Generating welcome greeting...", flush=True)
        welcome_msg = _generate_welcome_message(args)
        if welcome_msg:
            print(f"Avatar: {welcome_msg}")

    # Determine initial text: welcome message takes priority over --text
    if welcome_msg:
        _turn_start()
        build_queue.put(welcome_msg)
    elif initial_text:
        _turn_start()
        build_queue.put(initial_text)
    threading.Thread(target=input_loop, daemon=True).start()
    threading.Thread(target=builder_loop, daemon=True).start()

    pygame.init()
    try:
        try:
            fw0, fh0 = (int(v) for v in str(args.closeup_size).lower().split("x"))
        except ValueError:
            fw0, fh0 = (640, 640)
        screen = pygame.display.set_mode((min(560, fw0), min(560, fh0)), pygame.RESIZABLE)
        pygame.display.set_caption("PyToon - live lip-sync (type anything)")
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])
        font = None
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if hasattr(pygame, "font"):
                    try:
                        pygame.font.init()
                    except Exception:
                        pass
                    font = pygame.font.SysFont(None, 24)
        except Exception:
            font = None

        import collections
        idle_text = "Type a line in the terminal..."
        current = None  # (frames, proc, t0, n)
        pending: "collections.deque[tuple]" = collections.deque()
        idle_surf = None
        running = True
        pacing = time.monotonic()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if current is not None:
                            try:
                                current[1].terminate()
                            except Exception:
                                pass
                            current = None
                        pending.clear()
                        print("(stopped — cleared queue)")
                        _turn_end()  # Allow new input
                    elif event.key == pygame.K_q and (event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META)):
                        running = False

            # pick up newly built performances (non-blocking) -> FIFO queue
            try:
                while True:
                    # Check for interrupt first
                    if interrupt.is_set():
                        interrupt.clear()
                        if current is not None:
                            try:
                                current[1].terminate()
                            except Exception:
                                pass
                            current = None
                        pending.clear()
                        print("(interrupted)")
                        if not pending:
                            _turn_end()
                        break
                    item = play_queue.get_nowait()
                    if item is None:
                        running = False
                        break
                    pending.append(item)
            except Exception:
                pass
            if current is None and pending:
                frames, audio, seconds = pending.popleft()
                try:
                    f0 = frames[0]
                    disp_w = min(560, f0.shape[1])
                    disp_h = int(f0.shape[0] * (disp_w / f0.shape[1]) * args.scale)
                    if screen.get_size() != (disp_w, disp_h):
                        screen = pygame.display.set_mode((disp_w, disp_h), pygame.RESIZABLE)
                except Exception:
                    pass
                proc = play_audio(audio)
                current = (frames, proc, time.monotonic(), len(frames))
                idle_surf = None
                queued_more = f" (+{len(pending)} queued)" if pending else ""
                print(f"Playing {seconds:.1f}s{queued_more} (Esc stops, close window quits)...")

            w, h = screen.get_size()
            if current is not None:
                # Check for interrupt during playback
                if interrupt.is_set():
                    interrupt.clear()
                    try:
                        current[1].terminate()
                    except Exception:
                        pass
                    current = None
                    idle_surf = None
                    pending.clear()
                    print("(interrupted)")
                    _turn_end()
                else:
                    frames, proc, t0, n = current
                    elapsed = time.monotonic() - t0
                    idx = int(elapsed * args.fps)
                    if proc.poll() is not None and idx >= n:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        try:
                            idle_surf = pygame.image.frombuffer(
                                np.ascontiguousarray(frames[-1][:, :, :3]),
                                (frames[-1].shape[1], frames[-1].shape[0]), "RGB")
                        except Exception:
                            idle_surf = None
                        current = None
                        print("Done — type another line.\n")
                        # Signal that the full turn is complete (build + play)
                        if not pending:
                            _turn_end()
                    else:
                        idx = min(max(idx, 0), n - 1)
                        rgb = np.ascontiguousarray(frames[idx][:, :, :3])
                        surf = pygame.image.frombuffer(rgb, (rgb.shape[1], rgb.shape[0]), "RGB")
                        screen.blit(pygame.transform.smoothscale(surf, (w, h)), (0, 0))
                        pygame.display.flip()
            else:
                # Idle: show last frame static (no breathing/scaling)
                if idle_surf is not None:
                    try:
                        screen.fill((20, 20, 25))
                        # Center the idle frame without scaling
                        iw, ih = idle_surf.get_size()
                        screen.blit(idle_surf, ((w - iw) // 2, (h - ih) // 2))
                    except Exception:
                        screen.fill((20, 20, 25))
                else:
                    screen.fill((20, 20, 25))
                    if font is not None:
                        try:
                            msg = font.render(idle_text, True, (220, 220, 220))
                            screen.blit(msg, ((w - msg.get_width()) // 2,
                                             (h - msg.get_height()) // 2))
                        except Exception:
                            pass
                if building.is_set():
                    t = time.monotonic()
                    dots = "." * (1 + int(t * 2.5) % 3)
                    try:
                        if font is not None:
                            th = font.render("thinking" + dots, True, (140, 220, 140))
                            screen.blit(th, (12, h - th.get_height() - 10))
                        else:
                            r = 6 + int(3 * math.sin(t * 5))
                            pygame.draw.circle(screen, (140, 220, 140), (24, h - 24), max(2, r))
                    except Exception:
                        pass
                pygame.display.flip()

            while time.monotonic() - pacing < 1.0 / 30:
                time.sleep(0.005)
            pacing = time.monotonic()
    finally:
        stop.set()
        try:
            pygame.quit()
        except Exception:
            pass


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("text", nargs="?", help="Speak this first, then keep window open")
    p.add_argument("--text", dest="text_flag", help="Alternative to the positional text arg")
    p.add_argument("--fps", type=int, default=30, help="Playback frames per second")
    p.add_argument("--voice", default="AriaNeural",
                   help="Fixed TTS voice (locked to AriaNeural)")
    p.add_argument("--rate", type=int, default=None,
                   help="TTS speech rate (default: picked by mood, 180 if --no-mood)")
    p.add_argument("--edge-voice", default="en-US-AriaNeural",
                   help="Edge neural voice (locked to en-US-AriaNeural)")
    p.add_argument("--scale", type=float, default=1.0, help="Vertical headroom (playback only)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--closeup", action="store_true", help="Crop pytoon's avatar to just the face")
    mode.add_argument("--forward", action="store_true",
                      help="Use a custom front-facing cartoon face (always faces the viewer)")
    mode.add_argument("--closeup-size", default="640x640", help="Face resolution WxH (default 640x640)")
    p.add_argument("--cube", dest="cube", action="store_true", default=True,
                   help="Cube-bot avatar: white box with OLED mood eyes (default on)")
    p.add_argument("--no-cube", dest="cube", action="store_false",
                   help="Use the classic pytoon stickman avatar instead of the cube-bot")
    p.add_argument("--chat", action="store_true",
                   help="Reply with Ollama instead of repeating your text")
    p.add_argument("--ollama-model", default="llama3.2:1b",
                   help="Ollama model for --chat (default llama3.2:1b). Any locally installed Ollama model name/tag is accepted.")
    p.add_argument("--system", default=SYSTEM_PROMPT,
                   help="System prompt for --chat mode")
    p.add_argument("--memory-file", default=DEFAULT_HISTORY_FILE,
                   help="Where chat history is saved (default .tmp/chat_history.json)")
    p.add_argument("--facts-file", default=DEFAULT_FACTS_FILE,
                   help="Where long-term user facts are saved (default .tmp/user_facts.json)")
    p.add_argument("--reset-memory", action="store_true",
                   help="Erase saved history + facts and start fresh")
    p.add_argument("--memory-md", default=DEFAULT_MEMORY_MD,
                   help="The memory file: facts + summary of old chats (default .tmp/MEMORY.md)")
    p.add_argument("--compact-at", type=int, default=40,
                   help="Compact when history passes this many turns (default 40)")
    p.add_argument("--auto-facts", action="store_true",
                   help="Learn user facts in the background with an extra LLM call per turn (slower on CPU; SQLite remember/forget always stays on)")
    p.add_argument("--no-mood", action="store_true",
                   help="Disable mood-matched voices (use --voice/--rate or Samantha @180)")
    p.add_argument("--mood-llm", action="store_true",
                   help="Classify mood with Ollama instead of fast keywords (slower)")
    p.add_argument("--chibi", dest="chibi", action="store_true", default=False,
                   help="Chibi voice: raise pitch x1.25 (default off for a natural voice)")
    p.add_argument("--no-chibi", dest="chibi", action="store_false",
                   help="Disable the chibi pitch lift")
    p.add_argument("--pitch", type=float, default=None,
                   help="Custom pitch factor, e.g. 1.4 cuter, 0.8 deeper (implies chibi)")
    p.add_argument("--fast", action="store_true",
                   help="Way quicker: shorter Ollama replies (48 tokens), num_ctx 512, +10% speech rate, chunked playback, Pi-tuned threads")
    try:
        args = p.parse_args(argv)
    except SystemExit:
        raise
    except Exception:
        args = p.parse_args(argv)
    if args.fast:
        # in fast mode skip chibi pitch filter (saves an ffmpeg pass) unless --pitch given
        if args.pitch is None:
            args.no_chibi_fast = True
        else:
            args.no_chibi_fast = False
        # lower fps slightly in fast mode for less render work (optional)
        if args.fps == 30:
            args.fps = 26

    if args.chat:
        print(f"PyToon live + Ollama chat ({args.ollama_model}) — window stays open forever.")
        print("Type anything + Enter, the avatar replies out loud.")
    else:
        print("PyToon live — window stays open forever. Type anything + Enter.")
        print("Tip: add --chat to have Ollama reply instead of echoing you.")
    print("Type 'quit' or 'exit' to quit. Close the window to quit.\n")

    initial = args.text or args.text_flag
    try:
        run_persistent(args, initial_text=initial)
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
        sys.exit(130)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
        sys.exit(130)