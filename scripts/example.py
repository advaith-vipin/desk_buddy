#!/usr/bin/env python3
"""Example: lip-sync a talking face from an audio file + transcript.

Run:
    PYTHONPATH=src python scripts/example.py
"""

import os
import sys

if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desk_buddy._paths import ROOT, TMP_DIR
from pytoon.animator import animate

audio_file = os.path.join(TMP_DIR, "speech.mp3")
output = os.path.join(ROOT, "talking_face.output.mp4")

animation = animate(
    audio_file=audio_file,
    transcript="Hello, welcome to my application!",
)
animation.export(path=output)
print(f"Saved -> {output}")