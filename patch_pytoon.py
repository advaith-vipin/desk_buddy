"""Re-applies the moviepy-2 compatibility patch to the installed pytoon package.

pytoon 1.5.0 was written against moviepy 1.x (`moviepy.editor`) but pip now
resolves moviepy 2.x (required for Python 3.13 / numpy 2.x). This script
patches the installed pytoon/animator.py so the public API is unchanged:

    from pytoon.animator import animate
    animation = animate(audio_file="speech.mp3", transcript="...")
    animation.export(path="talking_face.output.mp4")

Usage (after creating a fresh venv + pip install pytoon):
    python patch_pytoon.py
"""

import os
import sys

PATCHES = [
    (
        "from moviepy.editor import ImageSequenceClip, CompositeVideoClip, CompositeAudioClip, AudioFileClip, VideoClip",
        "from moviepy import ImageSequenceClip, CompositeVideoClip, CompositeAudioClip, AudioFileClip, VideoClip, ImageClip",
    ),
    (
        "animation_clip = animation_clip.resize(width=new_width, height=new_height)",
        "animation_clip = animation_clip.resized(width=new_width, height=new_height)",
    ),
    (
        "audio_clip = CompositeAudioClip([audio_clip.set_start(0.2)])",
        "audio_clip = CompositeAudioClip([audio_clip.with_start(0.2)])",
    ),
    (
        "final_clip = final_clip.set_audio(audio_clip)",
        "final_clip = final_clip.with_audio(audio_clip)",
    ),
    (
        'animation_clip.set_position(("right", "bottom"))',
        'animation_clip.with_position(("right", "bottom"))',
    ),
]


def find_package() -> str:
    site_paths = []
    for base in sys.path:
        if base and base.endswith("site-packages") and os.path.isdir(base):
            site_paths.append(base)
    for base in site_paths:
        candidate = os.path.join(base, "pytoon", "animator.py")
        if os.path.exists(candidate):
            return candidate
    raise SystemExit("error: pytoon package not found in site-packages")


def patch() -> None:
    path = find_package()
    with open(path) as f:
        source = f.read()

    original = source
    for old, new in PATCHES:
        if old in source:
            source = source.replace(old, new)
            print(f"patched: {old[:60]}...")
        elif new in source:
            print(f"already patched: {old[:40]}...")
        else:
            print(f"SKIP (pattern not found): {old[:60]}...")

    if source == original:
        print("No changes needed.")
        return

    with open(path, "w") as f:
        f.write(source)
    print(f"Updated: {path}")


if __name__ == "__main__":
    patch()