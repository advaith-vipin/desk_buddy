#!/usr/bin/env bash
# Run script for desk_buddy (pytoon live avatar + CLI generator).
#
# Usage:
#   ./run.sh                        # live echo mode: type a line, avatar speaks it
#   ./run.sh --chat [--fast] ...     # live chat mode (Ollama replies out loud)
#   ./run.sh --cli --text "Hello"    # one-shot video render via pytoon_cli.py
#   ./run.sh --setup                 # install deps only, don't run anything
#   ./run.sh --help                  # this help (+ pytoon_live.py options)
#
# Any extra args (except --cli/--setup) are forwarded to pytoon_live.py.
# With --cli, extra args are forwarded to pytoon_cli.py.

set -euo pipefail

cd "$(dirname "$0")"

VENV_PY=".venv/bin/python"
SYS_PY="${PYTHON:-python3}"

msg()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \?//'
  echo
  if [ "$MODE" = "cli" ]; then
    echo "--- pytoon_cli.py options (CLI target) ---"
    "$PY" pytoon_cli.py --help
  else
    echo "--- pytoon_live.py options (default target) ---"
    "$PY" pytoon_live.py --help
  fi
}

# Resolve python: prefer project venv, fall back to system python3.
if [ -x "$VENV_PY" ]; then
  PY="$VENV_PY"
else
  command -v "$SYS_PY" >/dev/null 2>&1 || die "python3 not found. Install Python 3.10+ first."
  PY="$SYS_PY"
fi

setup_env() {
  # 1. Create venv if missing.
  if [ ! -x "$VENV_PY" ]; then
    msg "Creating virtualenv (.venv)..."
    "$SYS_PY" -m venv .venv
    PY="$VENV_PY"
  fi

  # 2. Install Python deps.
  if [ -f requirements.txt ]; then
    msg "Installing requirements..."
    "$PY" -m pip install --upgrade pip >/dev/null
    "$PY" -m pip install -r requirements.txt
  fi

  # 3. Patch installed pytoon for moviepy 2.x (idempotent).
  if [ -f patch_pytoon.py ]; then
    msg "Patching pytoon for moviepy 2.x (if needed)..."
    "$PY" patch_pytoon.py || warn "patch_pytoon.py failed; continuing anyway"
  fi

  mkdir -p .tmp

  # 4. System dependency checks (warnings only — TTS falls back gracefully).
  command -v ffmpeg >/dev/null 2>&1 || warn "ffmpeg not found; audio conversion will fail. Install it (sudo apt install ffmpeg)."
  if [ "$(uname)" != "Darwin" ]; then
    command -v espeak-ng >/dev/null 2>&1 || command -v espeak >/dev/null 2>&1 \
      || warn "espeak-ng not found; offline TTS fallback unavailable (sudo apt install espeak-ng)."
    command -v ffplay >/dev/null 2>&1 || command -v afplay >/dev/null 2>&1 \
      || warn "no audio player found (needs ffplay from ffmpeg, or afplay on macOS)."
  fi
}

MODE="live"
WANT_HELP=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help) WANT_HELP=1 ;;
    --setup)   MODE="setup" ;;
    --cli)     MODE="cli" ;;
    *)         ARGS+=("$arg") ;;
  esac
done

if [ "$WANT_HELP" -eq 1 ] && [ "$MODE" != "setup" ] && [ ${#ARGS[@]} -eq 0 ]; then
  usage; exit 0
elif [ "$WANT_HELP" -eq 1 ]; then
  ARGS+=("--help")  # e.g. ./run.sh --cli --help -> forward to the target script
fi

setup_env

case "$MODE" in
  setup)
    msg "Setup complete. Run './run.sh' or './run.sh --chat --fast' to start."
    ;;
  cli)
    [ -f pytoon_cli.py ] || die "pytoon_cli.py not found in $(pwd)"
    if [ ${#ARGS[@]} -eq 0 ]; then
      warn "no CLI args given; showing pytoon_cli.py --help"
      exec "$PY" pytoon_cli.py --help
    fi
    exec "$PY" pytoon_cli.py "${ARGS[@]}"
    ;;
  live)
    [ -f pytoon_live.py ] || die "pytoon_live.py not found in $(pwd)"
    if [[ " ${ARGS[*]} " == *" --chat "* ]] && ! command -v ollama >/dev/null 2>&1; then
      warn "ollama not found; --chat mode needs it (https://ollama.com). Continuing anyway."
    fi
    exec "$PY" pytoon_live.py "${ARGS[@]}"
    ;;
esac
