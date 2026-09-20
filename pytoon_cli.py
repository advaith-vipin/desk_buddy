"""Generate a lip-synced cartoon animation from a text script or an audio file.

Uses the pytoon library (Wav2Vec2 forced alignment + visemes) under the hood.

Examples:
    python pytoon_cli.py --text "Hello, welcome to my application!"
    python pytoon_cli.py --script script.txt --output out.mp4
    python pytoon_cli.py --audio speech.mp3 --transcript "audio words here"
    python pytoon_cli.py --audio speech.mp3                      # auto transcript
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(HERE, ".tmp")


def _ensure_venv() -> None:
    """Re-exec with the project venv python if pytoon is not importable.

    Lets users run `python pytoon_cli.py` with any Python and still land on
    the fully-installed environment.
    """
    try:
        import pytoon  # noqa: F401
        return
    except ImportError:
        pass

    venv_python = os.path.join(HERE, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        print(f"pytoon not available here; re-running with {venv_python}")
        os.execv(venv_python, [venv_python] + sys.argv)


_ensure_venv()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def synthesize_speech(text: str, out_mp3: str, voice: str | None = None,
                      rate: int = 180, pitch: float = 1.0) -> None:
    """Synthesizes text into an mp3 using the macOS 'say' TTS, then ffmpeg.

    pitch > 1.0 raises the voice (chibi effect) without changing speed.
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    script_file = os.path.join(TMP_DIR, "script.txt")
    aiff_file = os.path.join(TMP_DIR, "speech.aiff")
    with open(script_file, "w") as f:
        f.write(text)

    cmd = ["say", "-f", script_file, "-o", aiff_file, "-r", str(rate)]
    if voice:
        cmd += ["-v", voice]
    run(cmd)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", aiff_file]
    if pitch and abs(pitch - 1.0) > 1e-6:
        # raise pitch, then restore original tempo (duration unchanged)
        cmd += ["-filter:a", f"asetrate=22050*{pitch},aresample=22050,atempo={1.0 / pitch}"]
        print(f"(chibi pitch x{pitch})")
    cmd += ["-ac", "1", "-ar", "22050", "-codec:a", "libmp3lame", "-q:a", "4", out_mp3]
    run(cmd)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Text script to speak")
    src.add_argument("--script", help="Path to a text file with the script")
    src.add_argument("--audio", help="Existing audio file (.mp3/.wav) to lip-sync")

    p.add_argument("--transcript", help="Transcript of the audio (improves alignment accuracy)")
    p.add_argument("--output", default="talking_face.output.mp4", help="Output .mp4 path")
    p.add_argument("--fps", type=int, default=48, help="Video frames per second")
    p.add_argument("--voice", default=None,
                   help="Fixed macOS TTS voice (default: picked by mood, Samantha if --no-mood)")
    p.add_argument("--rate", type=int, default=None,
                   help="TTS speech rate in words-per-minute-ish (default: picked by mood, 180 if --no-mood)")
    p.add_argument("--background", help="Optional background video or image to overlay the avatar on")
    p.add_argument("--no-mood", action="store_true",
                   help="Disable mood-matched voices (use --voice/--rate or Samantha @180)")
    p.add_argument("--mood-llm", action="store_true",
                   help="Classify mood with Ollama instead of fast keywords (slower)")
    p.add_argument("--ollama-model", default="gemma3:4b",
                   help="Ollama model for --mood-llm (default gemma3:4b)")
    p.add_argument("--chibi", dest="chibi", action="store_true", default=True,
                   help="Chibi voice: raise pitch x1.25 (default on)")
    p.add_argument("--no-chibi", dest="chibi", action="store_false",
                   help="Disable the chibi pitch lift")
    p.add_argument("--pitch", type=float, default=None,
                   help="Custom pitch factor, e.g. 1.4 cuter, 0.8 deeper (implies chibi)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--closeup", action="store_true", help="Crop pytoon's avatar to just the face")
    mode.add_argument("--forward", action="store_true",
                      help="Use a custom front-facing cartoon face (always faces the viewer)")
    mode.add_argument("--closeup-size", default="720x720", help="Close-up resolution WxH (default 720x720)")
    return p.parse_args(argv)


MOOD_VOICES = {
    # mood: (macOS 'say' voice, speech rate) — brisk pacing
    "happy": ("Samantha", 215),
    "excited": ("Samantha", 225),
    "playful": ("Bubbles", 210),
    "silly": ("Bubbles", 210),
    "sad": ("Whisper", 160),
    "lonely": ("Whisper", 160),
    "angry": ("Zarvox", 190),
    "calm": ("Samantha", 175),
    "loving": ("Samantha", 175),
    "neutral": ("Samantha", 210),  # casual everyday chat
}

# voice used when the user tells the avatar to do something (and no strong
# emotion overrides it) — clear, attentive assistant tone
TASK_VOICE = ("Flo", 205)
# strong feelings keep their emotional voice even for commands
EMOTIONAL_MOODS = {"sad", "lonely", "angry", "excited", "playful", "silly", "loving"}

# checked in order — first match wins
MOOD_KEYWORDS = [
    ("angry", ["angry", "furious", "hate", "annoyed", "grr", "shut up", "rage"]),
    ("playful", ["haha", "lol", "hehe", "lmao", "joke", "funny", "silly", "boop"]),
    ("sad", ["sad", "sorry", "cry", "tears", "lonely", "heartbroken", "miss you"]),
    ("loving", ["love", "adore", "sweetheart", "darling", "honey", "kiss"]),
    ("excited", ["wow", "yay", "woohoo", "amazing", "awesome", "incredible", "!!"]),
    ("happy", ["happy", "glad", "joy", "wonderful", "congrats", "fantastic", "!"]),
    ("calm", ["calm", "relax", "sleep", "peaceful", "breathe", "good night"]),
]


def detect_mood(text: str, model: str | None = None) -> str:
    """Fast keyword mood detection; Ollama classifier if model is given."""
    if model:
        try:
            import ollama

            resp = ollama.chat(model=model, messages=[{
                "role": "user",
                "content": ("Classify the mood of this short reply into exactly one word: "
                            "happy, excited, playful, sad, angry, calm, loving, neutral. "
                            f"Reply with only that word.\n\nReply: {text!r}"),
            }])
            word = resp.message.content.strip().lower().split()[0].strip(".,!?\"'")
            if word in MOOD_VOICES:
                return word
        except Exception:
            pass  # fall through to keywords
    low = (text or "").lower()
    for mood, keywords in MOOD_KEYWORDS:
        if any(k in low for k in keywords):
            return mood
    return "neutral"


def resolve_voice(text: str, args, intent: str | None = None) -> tuple:
    """Picks (voice, rate) for this line. Explicit --voice/--rate always win."""
    if getattr(args, "no_mood", False):
        return (getattr(args, "voice", None) or "Samantha",
                getattr(args, "rate", None) or 200)
    model = getattr(args, "ollama_model", None) if getattr(args, "mood_llm", False) else None
    mood = detect_mood(text, model)
    intent = intent or detect_intent(text)
    if intent == "task" and mood not in EMOTIONAL_MOODS:
        mvoice, mrate = TASK_VOICE
    else:
        mvoice, mrate = MOOD_VOICES.get(mood, MOOD_VOICES["neutral"])
    voice = getattr(args, "voice", None) or mvoice
    rate = getattr(args, "rate", None) or mrate
    print(f"(intent: {intent}, mood: {mood} -> voice {voice} @ {rate}wpm)")
    return voice, rate


CHIBI_PITCH = 1.25  # default chibi lift


def resolve_pitch(args) -> float:
    """Chibi pitch factor. Explicit --pitch wins; --no-chibi disables."""
    if getattr(args, "pitch", None):
        return float(args.pitch)
    return CHIBI_PITCH if getattr(args, "chibi", True) else 1.0


CASUAL_PATTERNS = [
    "how are you", "how r you", "what's up", "whats up", "what is up",
    "what's happening", "whats happening", "what is happening",
    "how's it going",
    "hows it going", "how are things", "how do you feel", "you ok", "you okay",
    "good morning", "good afternoon", "good evening", "good night",
    "hi", "hello", "hey", "yo", "sup", "hiya", "heyy",
]

GREETING_OPENERS = {"hey", "heyy", "hi", "hiya", "hello", "yo", "sup", "oh", "well", "so"}

TASK_VERBS = [
    "tell", "remind", "set", "write", "explain", "describe", "list", "give",
    "show", "find", "search", "do", "make", "create", "help", "answer",
    "summarize", "summarise", "translate", "calculate", "play", "open",
    "remember", "plan", "check", "read", "sing", "say", "count",
]

TASK_PHRASES = [
    "please", "remind me", "don't forget", "dont forget", "can you", "could you",
    "would you", "will you", "do you know", "what is", "what are", "what was",
    "what were", "how do", "how does", "how did", "how can", "why is", "why are",
    "who is", "who are", "when is", "when are", "where is", "where are",
]


def detect_intent(text: str) -> str:
    """'task' if the user tells the avatar to do/answer something, else 'casual'."""
    low = (text or "").lower().strip().strip("!.,?").strip()
    if not low:
        return "casual"
    # strip leading "hey, hi, ..." so "hey, what's up" still reads as small talk
    toks = low.split()
    while toks and toks[0].strip(",!") in GREETING_OPENERS:
        toks.pop(0)
    core = " ".join(toks).strip(",! ").strip() or low
    has_q = "?" in (text or "")
    for pat in CASUAL_PATTERNS:  # pure small talk stays casual...
        if core == pat or (core.startswith(pat + " ") and len(core) < 40 and not has_q):
            return "casual"
    words = core.split()
    if words and words[0].strip(",!") in TASK_VERBS:
        return "task"
    if any(p in core for p in TASK_PHRASES):
        return "casual" if core in ("how are you",) else "task"
    return "casual"


def load_background(path: str, duration: float) -> object:
    from moviepy import VideoFileClip, ImageClip

    ext = path.rsplit(".", 1)[-1].lower()
    if ext in {"mp4", "mov", "avi", "mkv", "webm"}:
        return VideoFileClip(path)
    return ImageClip(path).with_duration(duration)


def closeup_frame(animation, idx: int, output_size=(720, 720), cache=None) -> object:
    """Crops and upscales one frame to a centered, chest-up face close-up."""
    from PIL import Image
    import numpy as np

    frames = animation.final_frames
    coords = animation.sequence.mouth_coords
    width, height = animation.frame_size
    out_w, out_h = output_size

    base = frames[idx]
    cw = min(int(out_h * (width / height)), width)
    mx, my = int(coords[idx].x), int(coords[idx].y)
    left = min(max(mx - cw // 2, 0), width - cw)
    top = min(max(my - int(cw * 0.62), 0), height - cw)

    key = (idx, left, top, cw, out_w, out_h)
    if cache is not None and cache.get("key") == key:
        return cache["frame"]

    face = np.asarray(Image.fromarray(base[top : top + cw, left : left + cw]).resize((out_w, out_h), Image.LANCZOS))
    if cache is not None:
        cache["key"] = key
        cache["frame"] = face
    return face


def export_closeup(animation, path: str, output_size=(720, 720)) -> None:
    """Exports a face-only close-up mp4 (no background passed = defaults)."""
    from moviepy import CompositeVideoClip, CompositeAudioClip, AudioFileClip, VideoClip, ColorClip

    def default_background(size=(720, 720)):
        return ColorClip(size=size, color=(135, 206, 235))

    out_w, out_h = output_size
    n = len(animation.final_frames)
    duration = n / animation.fps
    cache = {}

    def make_frame(t):
        idx = min(int(t * animation.fps), n - 1)
        return closeup_frame(animation, idx, output_size, cache)[:, :, :3]

    def make_mask(t):
        idx = min(int(t * animation.fps), n - 1)
        alpha = closeup_frame(animation, idx, output_size, cache)[:, :, 3]
        return alpha.astype("float32") / 255.0

    clip = VideoClip(make_frame, duration=duration)
    clip.size = (out_w, out_h)
    mask = VideoClip(make_mask, is_mask=True, duration=duration)
    mask.size = (out_w, out_h)
    clip = clip.with_mask(mask)

    background = default_background(size=(out_w, out_h)).with_duration(duration)
    final_clip = CompositeVideoClip(
        clips=[background, clip.with_position("center")], use_bgclip=True
    )

    audio_clip = AudioFileClip(animation.audio_file)
    audio_clip = CompositeAudioClip([audio_clip.with_start(0.2)])
    final_clip = final_clip.with_audio(audio_clip)

    final_clip.write_videofile(
        path, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, fps=animation.fps
    )


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(TMP_DIR, exist_ok=True)

    if args.text or args.script:
        if args.script:
            with open(args.script) as f:
                text = f.read().strip()
        else:
            text = args.text.strip()
        if not text:
            sys.exit("error: empty script")

        audio_file = os.path.join(TMP_DIR, "speech.mp3")
        voice, rate = resolve_voice(text, args)
        synthesize_speech(text, audio_file, voice=voice, rate=rate,
                          pitch=resolve_pitch(args))
        print(f"Synthesized speech -> {audio_file}")
        transcript = args.transcript if args.transcript else text
    else:
        audio_file = os.path.abspath(args.audio)
        if not os.path.exists(audio_file):
            sys.exit(f"error: audio file not found: {audio_file}")
        transcript = args.transcript or None
        print(f"Using audio -> {audio_file}")

    print("Aligning phonemes and building mouth sequence (downloads Wav2Vec2 on first run)...")
    from pytoon.animator import animate

    if args.forward:
        from front_face import forward_export

        try:
            out_w, out_h = (int(v) for v in str(args.closeup_size).lower().split("x"))
        except ValueError:
            sys.exit(f"error: --closeup-size must be WxH, got {args.closeup_size!r}")
        forward_export(audio_file, args.output, transcript, args.fps, (out_w, out_h))
    else:
        animation = animate(audio_file=audio_file, transcript=transcript, fps=args.fps)

        if args.closeup:
            try:
                out_w, out_h = (int(v) for v in str(args.closeup_size).lower().split("x"))
            except ValueError:
                sys.exit(f"error: --closeup-size must be WxH, got {args.closeup_size!r}")
            export_closeup(animation, args.output, output_size=(out_w, out_h))
        elif args.background:
            background = load_background(args.background, animation.duration)
            animation.export(path=args.output, background=background)
        else:
            from moviepy import ColorClip
            background = ColorClip(size=animation.frame_size, color=(135, 206, 235)).with_duration(animation.duration)
            animation.export(path=args.output, background=background)

    print(f"Saved animation -> {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()