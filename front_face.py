"""A front-facing cartoon talking head rendered with Pillow.

The face always faces the viewer (symmetric, eyes on camera). Its mouth is
driven by pytoon's forced-aligned viseme sequence, so lip timing still comes
from the same Wav2Vec2 audio alignment used by pytoon.

Exports do NOT hold all frames in memory: frames are rendered lazily on demand
during video encoding.
"""

import os

import numpy as np
from PIL import Image, ImageDraw

import pytoon
from pytoon.lipsync import viseme_sequencer

_VISEME_DIR = os.path.join(os.path.dirname(pytoon.__file__), "assets", "visemes", "positive")
_MOUTH_CACHE = {}
_FRAME_CACHE = {}

# Palette
SKIN = (255, 222, 186)
SKIN_SHADE = (238, 199, 160)
HAIR = (74, 47, 29)
SHIRT = (59, 130, 246)
SHIRT_DARK = (43, 105, 205)
WHITE = (255, 255, 255)
PUPIL = (43, 32, 24)
BLUSH = (255, 168, 168)

# Head geometry (720x720 canvas)
HEAD_BOX = (170, 60, 550, 520)          # cx=360, cy=290, rx=190, ry=230
EAR_L = (175, 272, 235, 348)
EAR_R = (485, 272, 545, 348)
EYE_L = (265, 254, 355, 346)            # cx=310
EYE_R = (365, 254, 455, 346)            # cx=410
MOUTH_CENTER = (360, 432)
MOUTH_SIZE = (176, 88)


def _load_mouth(name: str) -> Image.Image:
    img = _MOUTH_CACHE.get(name)
    if img is None:
        img = Image.open(os.path.join(_VISEME_DIR, name)).convert("RGBA")
        _MOUTH_CACHE[name] = img
    return img


def eye_openness(frame_index: float, fps: int = 48) -> float:
    """1.0 = open, 0.0 = shut. Blinks every ~3s."""
    t = (frame_index / fps) % 3.0
    return float(np.clip(1.0 - abs(t - 2.93) / 0.16, 0.0, 1.0))


def render_forward_frame(viseme_name: str, frame_index: int, fps: int = 48,
                         size: tuple = (720, 720)) -> np.ndarray:
    """Renders one front-facing cartoon frame with the given viseme mouth."""
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)

    # Torso / shoulders
    d.rounded_rectangle((150, 545, 570, size[1] + 10), radius=90, fill=SHIRT)
    d.polygon([(285, 720), (435, 720), (400, 585), (320, 585)], fill=SHIRT_DARK)

    # Neck
    d.rounded_rectangle((325, 455, 395, 610), radius=28, fill=SKIN)
    d.rounded_rectangle((350, 455, 370, 610), radius=10, fill=SKIN_SHADE)

    # Head
    d.ellipse(HEAD_BOX, fill=SKIN)

    # Hair: cap over the top of the head + sideburns
    d.pieslice((140, 5, 580, 305), start=180, end=360, fill=HAIR)
    d.rounded_rectangle((172, 250, 210, 340), radius=6, fill=HAIR)
    d.rounded_rectangle((510, 250, 548, 340), radius=6, fill=HAIR)

    # Ears
    d.ellipse(EAR_L, fill=SKIN)
    d.ellipse(EAR_R, fill=SKIN)
    d.ellipse(EAR_L, fill=SKIN_SHADE)

    face_y0, face_y1 = HEAD_BOX[1], HEAD_BOX[3]

    # Eyebrows
    d.line([(272, 214), (312, 204)], fill=HAIR, width=11)
    d.line([(408, 204), (448, 214)], fill=HAIR, width=11)

    open_ = eye_openness(frame_index, fps)

    # Eyes (symmetric, looking at the viewer)
    for box, cx in ((EYE_L, 310), (EYE_R, 410)):
        cy = (box[1] + box[3]) / 2
        ry = (box[3] - box[1]) / 2 * open_
        eye_ry = max(ry, 2.0)
        white = [box[0], int(cy - eye_ry), box[2], int(cy + eye_ry)]
        d.ellipse(white, fill=WHITE)
        d.ellipse((cx - 17, int(cy) - 17, cx + 17, int(cy) + 17), fill=PUPIL)
        d.ellipse((cx - 8, int(cy) - 9, cx - 1, int(cy) - 2), fill=WHITE)
        # eyelid line when partially/fully closed
        d.line([(box[0] + 4, int(cy)), (box[2] - 4, int(cy))], fill=(225, 185, 148), width=3)

    # Nose (subtle)
    d.arc((342, 330, 378, 372), start=0, end=180, fill=SKIN_SHADE, width=5)

    # Blush
    blush = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(blush).ellipse((278, 396, 328, 432), fill=BLUSH + (110,))
    ImageDraw.Draw(blush).ellipse((392, 396, 442, 432), fill=BLUSH + (110,))
    canvas.alpha_composite(blush, (0, 0))

    # Mouth (driven by the aligned viseme)
    mouth = _load_mouth(viseme_name).resize(MOUTH_SIZE, Image.LANCZOS)
    canvas.alpha_composite(mouth, (MOUTH_CENTER[0] - MOUTH_SIZE[0] // 2,
                                   MOUTH_CENTER[1] - MOUTH_SIZE[1] // 2))

    chin = face_y1
    d = ImageDraw.Draw(canvas)  # refresh draw so mouth overlaps cleanly
    # jaw shadow under the mouth
    shadow = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse((300, chin - 26, 420, chin + 6), fill=(0, 0, 0, 40))
    canvas.alpha_composite(shadow, (0, 0))
    _ = d

    return np.asarray(canvas)


def viseme_frames_for(audio_file: str, transcript: str = None, fps: int = 48) -> list:
    """Runs Wav2Vec2 alignment and returns the flat per-frame viseme list."""
    sequence = viseme_sequencer(audio_file=audio_file, transcript=transcript, fps=fps)
    frames = []
    for ws in sequence:
        if ws.visemes:
            frames.extend(ws.visemes)
    return frames


def forward_export(audio_file: str, path: str, transcript: str = None, fps: int = 48,
                   size: tuple = (720, 720), background=None):
    """Exports the front-facing animation to an .mp4 with audio, lazy frames."""
    from moviepy import CompositeVideoClip, CompositeAudioClip, AudioFileClip, VideoClip, ColorClip

    def default_background(size=(720, 720)):
        return ColorClip(size=size, color=(135, 206, 235))

    visemes = viseme_frames_for(audio_file, transcript, fps)
    n = len(visemes)
    duration = n / fps
    cache = {}

    def make_frame(t):
        idx = min(int(t * fps), n - 1)
        key = (idx,)
        frame = cache.get("frame") if cache.get("key") == key else None
        if frame is None:
            frame = render_forward_frame(visemes[idx], idx, fps, size)
            cache["key"] = key
            cache["frame"] = frame
        return frame[:, :, :3]

    def make_mask(t):
        idx = min(int(t * fps), n - 1)
        key = (idx,)
        frame = cache.get("frame") if cache.get("key") == key else None
        if frame is None:
            frame = render_forward_frame(visemes[idx], idx, fps, size)
            cache["key"] = key
            cache["frame"] = frame
        return frame[:, :, 3].astype("float32") / 255.0

    clip = VideoClip(make_frame, duration=duration)
    clip.size = size
    mask = VideoClip(make_mask, is_mask=True, duration=duration)
    mask.size = size
    clip = clip.with_mask(mask)

    if background is None:
        background = default_background(size=size).with_duration(duration)

    final_clip = CompositeVideoClip(
        clips=[background, clip.with_position("center")], use_bgclip=True
    )

    audio_clip = CompositeAudioClip([AudioFileClip(audio_file).with_start(0.2)])
    final_clip = final_clip.with_audio(audio_clip)

    final_clip.write_videofile(
        path, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, fps=fps
    )
    return visemes