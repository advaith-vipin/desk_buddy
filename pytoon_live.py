"""Live pygame player: type a line, watch a cartoon lip-sync it instantly.

Interactive mode:
    python pytoon_live.py

Chat mode (Ollama replies, avatar speaks the reply):
    python pytoon_live.py --chat
    python pytoon_live.py --chat --ollama-model gemma3:4b

One-shot mode:
    python pytoon_live.py "Hello, welcome to my application!"

Per line it: synthesizes speech (macOS 'say'), force-aligns phonemes with
Wav2Vec2, builds the cartoon mouth/pose frames, then plays frames in a window
in real time while the audio plays through the speakers. Closing the window
(or pressing Ctrl-C) stops playback. The window stays open forever until you
type quit/exit or close it.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(HERE, ".tmp")


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
                      pitch: float = 1.0) -> str:
    """Synthesizes text to an mp3 and returns its path.

    pitch > 1.0 raises the voice (chibi effect) without changing speed.
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    script_file = os.path.join(TMP_DIR, "live_script.txt")
    aiff_file = os.path.join(TMP_DIR, "live_speech.aiff")
    mp3_file = os.path.join(TMP_DIR, "live_speech.mp3")

    with open(script_file, "w") as f:
        f.write(text)

    cmd = ["say", "-f", script_file, "-o", aiff_file, "-r", str(rate)]
    if voice:
        cmd += ["-v", voice]
    subprocess.run(cmd, check=True)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", aiff_file]
    if pitch and abs(pitch - 1.0) > 1e-6:
        # raise pitch, then restore original tempo (duration unchanged)
        cmd += ["-filter:a", f"asetrate=22050*{pitch},aresample=22050,atempo={1.0 / pitch}"]
        print(f"(chibi pitch x{pitch})")
    cmd += ["-ac", "1", "-ar", "22050", "-codec:a", "libmp3lame", "-q:a", "4", mp3_file]
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


def chat_reply(user_text: str, history: list, model: str) -> str:
    """Asks Ollama for a short spoken reply. Updates history in place."""
    import ollama

    history.append({"role": "user", "content": user_text})
    # keep memory bounded: system msg + last 20 turns
    short_history = ([history[0]] + history[-20:]) if history and history[0].get("role") == "system" else history[-20:]
    # num_predict caps reply length -> quicker answers AND quicker speech
    resp = ollama.chat(model=model, messages=short_history, options={"num_predict": 80})
    reply = resp.message.content.strip()
    history.append({"role": "assistant", "content": reply})
    return reply


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
            resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
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
    parts = [base_system]
    if facts:
        parts.append("Things you remember about the user:\n" + "\n".join(f"- {f}" for f in facts))
    if summary:
        parts.append("Summary of older conversations (already forgotten verbatim, remember the gist):\n" + summary)
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
        resp = ollama.chat(model=args.ollama_model, messages=[{"role": "user", "content": prompt}])
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


def build_frames_for_line(line: str, args, intent: str | None = None):
    """Synthesizes + aligns one line. Returns (frames, audio_path, seconds)."""
    from pytoon.animator import animate
    from pytoon_cli import resolve_voice, resolve_pitch

    voice, rate = resolve_voice(line, args, intent)
    audio = synthesize_speech(line, voice=voice, rate=rate, pitch=resolve_pitch(args))
    if args.forward:
        from front_face import viseme_frames_for, render_forward_frame

        try:
            fw, fh = (int(v) for v in str(args.closeup_size).lower().split("x"))
        except ValueError:
            sys.exit(f"error: --closeup-size must be WxH, got {args.closeup_size!r}")
        print("Aligning phonemes and building front-facing frames...")
        visemes = viseme_frames_for(audio, line, args.fps)
        frames = [render_forward_frame(v, i, args.fps, (fw, fh)) for i, v in enumerate(visemes)]
        seconds = len(frames) / args.fps
    else:
        print("Aligning phonemes and building frames...")
        animation = animate(audio_file=audio, transcript=line, fps=args.fps)
        frames = animation.final_frames
        if args.closeup:
            from pytoon_cli import closeup_frame

            try:
                cw, ch = (int(v) for v in str(args.closeup_size).lower().split("x"))
            except ValueError:
                sys.exit(f"error: --closeup-size must be WxH, got {args.closeup_size!r}")
            print("Cropping to face close-up...")
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

    def input_loop():
        while not stop.is_set():
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
                    from pytoon_cli import detect_intent

                    intent = detect_intent(line)  # intent comes from YOUR words
                    mem_now = load_memory_md(mem_path)  # pick up background-learned facts
                    facts_now = mem_now["facts"] or load_facts(facts_path)
                    if facts_now != (mem_now["facts"] or []) and facts_now:
                        save_memory_md(mem_path, facts_now, mem_now["summary"])
                        mem_now["facts"] = facts_now
                    inject_memory(history, args.system, facts_now, mem_now["summary"])
                    print("Thinking (Ollama)...")
                    speak = chat_reply(line, history, args.ollama_model)
                    print(f"Avatar: {speak}")
                    save_history(getattr(args, "memory_file", None) or DEFAULT_HISTORY_FILE, history)
                    refresh_facts_background(line, speak, facts_path, args.ollama_model)
                    compact_memory(history, args)  # rare: summarizes old turns into MEMORY.md
                print("Synthesizing speech...")
                frames, audio, seconds = build_frames_for_line(speak, args, intent)
                print(f"Queued {seconds:.1f}s — playing next...")
                play_queue.put((frames, audio, seconds))
            except Exception as e:  # keep window alive on errors
                print(f"error building animation: {e}\n")
            finally:
                building.clear()

    def warmup_loop():
        """Preloads Ollama + speech models in the background so the first
        reply is quicker (everything is cached for later turns)."""
        try:
            if getattr(args, "chat", False):
                import ollama

                ollama.chat(model=args.ollama_model,
                            messages=[{"role": "user", "content": "hi"}],
                            options={"num_predict": 1})
        except Exception:
            pass
        try:
            from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

            WAV2VEC2_ASR_BASE_960H.get_model()
        except Exception:
            pass
        if not stop.is_set():
            print("(warmed up — replies will be quicker)")

    if initial_text:
        build_queue.put(initial_text)
    threading.Thread(target=input_loop, daemon=True).start()
    threading.Thread(target=builder_loop, daemon=True).start()
    threading.Thread(target=warmup_loop, daemon=True).start()

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

        idle_text = "Type a line in the terminal..."
        current = None  # (frames, proc, t0, n)
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
                    elif event.key == pygame.K_q and (event.mod & pygame.KMOD_CTRL):
                        running = False

            # pick up newly built performances (non-blocking)
            try:
                while True:
                    item = play_queue.get_nowait()
                    if item is None:
                        running = False
                        break
                    frames, audio, seconds = item
                    if current is not None:
                        try:
                            current[1].terminate()
                        except Exception:
                            pass
                    # resize window to new frames on first play
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
                    print(f"Playing {seconds:.1f}s (Esc stops, close window quits)...")
            except Exception:
                pass

            w, h = screen.get_size()
            if current is not None:
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
                else:
                    idx = min(max(idx, 0), n - 1)
                    rgb = np.ascontiguousarray(frames[idx][:, :, :3])
                    surf = pygame.image.frombuffer(rgb, (rgb.shape[1], rgb.shape[0]), "RGB")
                    screen.blit(pygame.transform.smoothscale(surf, (w, h)), (0, 0))
                    pygame.display.flip()
            else:
                # idle animation: gentle breathing (zoom) + sway, so the avatar
                # feels alive while waiting; animated dots while building
                import math

                t = time.monotonic()
                breath = 1.0 + 0.025 * math.sin(t * 1.6)
                dx = int(8 * math.sin(t * 0.9))
                if idle_surf is not None:
                    try:
                        iw, ih = max(1, int(w * breath)), max(1, int(h * breath))
                        big = pygame.transform.smoothscale(idle_surf, (iw, ih))
                        screen.fill((20, 20, 25))
                        screen.blit(big, ((w - iw) // 2 + dx, (h - ih) // 2))
                    except Exception:
                        screen.fill((20, 20, 25))
                else:
                    screen.fill((20, 20, 25))
                    if font is not None:
                        try:
                            msg = font.render(idle_text, True, (220, 220, 220))
                            bob = int(4 * math.sin(t * 1.6))
                            screen.blit(msg, ((w - msg.get_width()) // 2,
                                             (h - msg.get_height()) // 2 + bob))
                        except Exception:
                            pass
                if building.is_set():
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
        pygame.quit()


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("text", nargs="?", help="Speak this first, then keep window open")
    p.add_argument("--text", dest="text_flag", help="Alternative to the positional text arg")
    p.add_argument("--fps", type=int, default=30, help="Playback frames per second")
    p.add_argument("--voice", default=None,
                   help="Fixed macOS TTS voice (default: picked by mood, Samantha if --no-mood; try Bubbles)")
    p.add_argument("--rate", type=int, default=None,
                   help="TTS speech rate (default: picked by mood, 180 if --no-mood)")
    p.add_argument("--scale", type=float, default=1.0, help="Vertical headroom (playback only)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--closeup", action="store_true", help="Crop pytoon's avatar to just the face")
    mode.add_argument("--forward", action="store_true",
                      help="Use a custom front-facing cartoon face (always faces the viewer)")
    mode.add_argument("--closeup-size", default="640x640", help="Face resolution WxH (default 640x640)")
    p.add_argument("--chat", action="store_true",
                   help="Reply with Ollama instead of repeating your text")
    p.add_argument("--ollama-model", default="gemma3:4b",
                   help="Ollama model for --chat (default gemma3:4b)")
    p.add_argument("--system", default=(
        "You are a friendly cartoon character talking to the user. "
        "Keep replies very short: 1 to 2 sentences. Plain text only, no markdown, "
        "no emojis, no stage directions. Chat casually like a friend for small "
        "talk; when the user asks you to do something or asks a question, "
        "just answer helpfully and directly."),
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
    p.add_argument("--no-mood", action="store_true",
                   help="Disable mood-matched voices (use --voice/--rate or Samantha @180)")
    p.add_argument("--mood-llm", action="store_true",
                   help="Classify mood with Ollama instead of fast keywords (slower)")
    p.add_argument("--chibi", dest="chibi", action="store_true", default=True,
                   help="Chibi voice: raise pitch x1.25 (default on)")
    p.add_argument("--no-chibi", dest="chibi", action="store_false",
                   help="Disable the chibi pitch lift")
    p.add_argument("--pitch", type=float, default=None,
                   help="Custom pitch factor, e.g. 1.4 cuter, 0.8 deeper (implies chibi)")
    args = p.parse_args(argv)

    if args.chat:
        print(f"PyToon live + Ollama chat ({args.ollama_model}) — window stays open forever.")
        print("Type anything + Enter, the avatar replies out loud.")
    else:
        print("PyToon live — window stays open forever. Type anything + Enter.")
        print("Tip: add --chat to have Ollama reply instead of echoing you.")
    print("Type 'quit' or 'exit' to quit. Close the window to quit.\n")

    initial = args.text or args.text_flag
    run_persistent(args, initial_text=initial)


if __name__ == "__main__":
    main()