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
    "$PY" -m desk_buddy.entrypoints.pytoon_cli --help
  else
    echo "--- pytoon_live.py options (default target) ---"
    "$PY" -m desk_buddy.entrypoints.pytoon_live --help
  fi
}

# Resolve python: prefer project venv, fall back to system python3.
if [ -x "$VENV_PY" ]; then
  PY="$VENV_PY"
else
  command -v "$SYS_PY" >/dev/null 2>&1 || die "python3 not found. Install Python 3.10+ first."
  PY="$SYS_PY"
fi

# Make the src/ package importable (desk_buddy.*)
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
export PYTHONDONTWRITEBYTECODE=1

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
INTERACTIVE=0

# If no arguments provided, run in interactive mode
if [ $# -eq 0 ]; then
  INTERACTIVE=1
else
  for arg in "$@"; do
    case "$arg" in
      -h|--help) WANT_HELP=1 ;;
      --setup)   MODE="setup" ;;
      --cli)     MODE="cli" ;;
      --chat)    ARGS+=("--chat") ;;
      --fast)    ARGS+=("--fast") ;;
      --non-interactive) INTERACTIVE=0 ;;
      *)         ARGS+=("$arg") ;;
    esac
  done
fi

# Interactive parameter selection
if [ "$INTERACTIVE" -eq 1 ]; then
  clear 2>/dev/null || true
  echo "======================================"
  echo "  Desk Buddy - Interactive Launcher"
  echo "======================================"
  echo

  # Mode selection
  PS3="Select a mode [1-6]: "
  COLUMNS=1
  options=("Live Echo (type and avatar speaks)" "Live Chat (AI replies out loud)" "CLI Video (one-shot render)" "Setup only (install dependencies)" "Show help" "Quit")
  select opt in "${options[@]}"; do
    case "$REPLY" in
      1)
        MODE="live"
        break
        ;;
      2)
        MODE="live"
        ARGS+=("--chat")
        break
        ;;
      3)
        MODE="cli"
        break
        ;;
      4)
        MODE="setup"
        break
        ;;
      5)
        usage
        exit 0
        ;;
      6)
        echo "Goodbye!"
        exit 0
        ;;
      *)
        echo "Invalid option. Please select 1-6."
        ;;
    esac
  done
  echo

  # Mode-specific prompts
  case "$MODE" in
    live)
      # Check if chat mode was selected
      if [[ " ${ARGS[*]} " == *" --chat "* ]]; then
        echo "--- Chat Mode Options ---"
        read -r -p "Use fast mode? (shorter replies, faster on CPU) [y/N]: " fast_choice
        if [[ "$fast_choice" =~ ^[Yy]$ ]]; then
          ARGS+=("--fast")
        fi

        echo
        # Pull available Ollama models
        AVAILABLE_MODELS=()
        if command -v ollama >/dev/null 2>&1; then
          while IFS= read -r line; do
            [ -z "$line" ] && continue
            AVAILABLE_MODELS+=("$line")
          done < <(ollama list 2>&1 | awk 'NR>1 {print $1}' | grep -v '^NAME$')
        fi

        if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
          echo "No Ollama models detected or 'ollama' not found."
          read -r -p "Enter model name manually (e.g. llama3.2:1b): " manual_model
          if [ -n "$manual_model" ]; then
            if command -v ollama >/dev/null 2>&1; then
              echo "Pulling $manual_model..."
              ollama pull "$manual_model" || warn "Failed to pull $manual_model; continuing anyway"
            fi
            ARGS+=("--ollama-model" "$manual_model")
          fi
        else
          echo "Ensuring Ollama models are available (pulling if needed)..."
          if command -v ollama >/dev/null 2>&1; then
            for m in "${AVAILABLE_MODELS[@]}"; do
              echo "  Pulling $m..."
              ollama pull "$m" >/dev/null 2>&1 || true
            done
          fi

          echo
          echo "Available Ollama models on this system:"
          for i in "${!AVAILABLE_MODELS[@]}"; do
            echo "  $((i+1))) ${AVAILABLE_MODELS[$i]}"
          done
          echo "  $((i+2))) Enter custom model name"

          read -r -p "Select model [1-$((i+2))] (default: 1): " model_choice
          if [ -z "$model_choice" ] || [ "$model_choice" = "1" ]; then
            selected="${AVAILABLE_MODELS[0]}"
            if command -v ollama >/dev/null 2>&1; then
              echo "Pulling $selected..."
              ollama pull "$selected" >/dev/null 2>&1 || true
            fi
            ARGS+=("--ollama-model" "$selected")
          elif [[ "$model_choice" =~ ^[0-9]+$ ]] && [ "$model_choice" -ge 2 ] && [ "$model_choice" -le $((i+1)) ]; then
            idx=$((model_choice-1))
            selected="${AVAILABLE_MODELS[$idx]}"
            if command -v ollama >/dev/null 2>&1; then
              echo "Pulling $selected..."
              ollama pull "$selected" >/dev/null 2>&1 || true
            fi
            ARGS+=("--ollama-model" "$selected")
          elif [[ "$model_choice" =~ ^[0-9]+$ ]] && [ "$model_choice" -eq $((i+2)) ]; then
            read -r -p "Custom model name: " custom_model
            if [ -n "$custom_model" ]; then
              if command -v ollama >/dev/null 2>&1; then
                echo "Pulling $custom_model..."
                ollama pull "$custom_model" || warn "Failed to pull $custom_model; continuing anyway"
              fi
              ARGS+=("--ollama-model" "$custom_model")
            fi
          else
            selected="${AVAILABLE_MODELS[0]}"
            if command -v ollama >/dev/null 2>&1; then
              echo "Pulling $selected..."
              ollama pull "$selected" >/dev/null 2>&1 || true
            fi
            ARGS+=("--ollama-model" "$selected")
          fi
        fi

        read -r -p "Reset chat memory? (starts fresh) [y/N]: " reset_choice
        if [[ "$reset_choice" =~ ^[Yy]$ ]]; then
          ARGS+=("--reset-memory")
        fi
        echo
      else
        echo "--- Live Echo Mode ---"
        read -r -p "Optional text to speak on startup (leave empty to type interactively): " startup_text
        if [ -n "$startup_text" ]; then
          ARGS+=("$startup_text")
        fi
        echo
      fi
      ;;
    cli)
      echo "--- CLI Video Mode ---"
      while true; do
        read -r -p "Enter text to render as video (required): " cli_text
        if [ -n "$cli_text" ]; then
          ARGS+=("--text" "$cli_text")
          break
        else
          echo "Text is required for CLI mode."
        fi
      done

      read -r -p "Output filename (optional, e.g. output.mp4): " cli_out
      if [ -n "$cli_out" ]; then
        ARGS+=("--out" "$cli_out")
      fi

      read -r -p "Voice style (cube/forward/closeup)? [cube/forward/closeup/N]: " cli_style
      case "$cli_style" in
        cube|Cube) ARGS+=("--cube") ;;
        forward|Forward) ARGS+=("--forward") ;;
        closeup|Closeup) ARGS+=("--closeup") ;;
      esac
      echo
      ;;
  esac

  echo "Starting with: Mode=$MODE Args=(${ARGS[*]-none})"
  echo "======================================"
  sleep 1
fi

if [ "$WANT_HELP" -eq 1 ] && [ "$MODE" != "setup" ] && [ ${#ARGS[@]} -eq 0 ]; then
  usage; exit 0
elif [ "$WANT_HELP" -eq 1 ]; then
  ARGS+=("--help")  # e.g. ./run.sh --cli --help -> forward to the target script
fi

# Suppress macOS duplicate dylib warnings (SDL/FFmpeg from pygame/cv2 vs system)
if [ "$(uname)" = "Darwin" ]; then
  export DYLD_LIBRARY_PATH=""
  if [ -d ".venv/lib/python3.13/site-packages/pygame/.dylibs" ]; then
    export DYLD_LIBRARY_PATH=".venv/lib/python3.13/site-packages/pygame/.dylibs:$DYLD_LIBRARY_PATH"
  fi
  if [ -d ".venv/lib/python3.13/site-packages/cv2/.dylibs" ]; then
    export DYLD_LIBRARY_PATH=".venv/lib/python3.13/site-packages/cv2/.dylibs:$DYLD_LIBRARY_PATH"
  fi
fi

setup_env

case "$MODE" in
  setup)
    msg "Setup complete. Run './run.sh' or './run.sh --chat --fast' to start."
    ;;
  cli)
    [ -f src/desk_buddy/entrypoints/pytoon_cli.py ] || die "pytoon_cli.py not found in $(pwd)/src/desk_buddy"
    if [ ${#ARGS[@]} -eq 0 ]; then
      warn "no CLI args given; showing pytoon_cli.py --help"
      exec "$PY" -m desk_buddy.entrypoints.pytoon_cli --help
    fi
    exec "$PY" -m desk_buddy.entrypoints.pytoon_cli "${ARGS[@]}"
    ;;
  live)
    [ -f src/desk_buddy/entrypoints/pytoon_live.py ] || die "pytoon_live.py not found in $(pwd)/src/desk_buddy"
    if [[ " ${ARGS[*]} " == *" --chat "* ]] && ! command -v ollama >/dev/null 2>&1; then
      warn "ollama not found; --chat mode needs it (https://ollama.com). Continuing anyway."
    fi
    exec "$PY" -m desk_buddy.entrypoints.pytoon_live "${ARGS[@]}"
    ;;
esac
