import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAMPLES = 16
CANDIDATES = 4
TILE = 384


@dataclass
class Frame:
    index: int
    image: Image.Image
    score: float


def sharpness(gray: np.ndarray) -> float:
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def frame_score(rgb: np.ndarray) -> float:
    gray = rgb.astype(np.float32).mean(axis=2)
    brightness = float(gray.mean())
    exposure = 1.0 - abs(brightness - 128.0) / 128.0
    if brightness < 20 or brightness > 240:
        exposure *= 0.1
    return sharpness(gray) * max(exposure, 0.01)


def sample_frames(path: Path, samples: int = SAMPLES) -> list[Frame]:
    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        total = reader.count_frames()
        if total <= 0:
            return []
        step = max(1, total // samples)
        frames: list[Frame] = []
        for index in range(0, total, step):
            rgb = np.asarray(reader.get_data(index))[:, :, :3]
            frames.append(Frame(index, Image.fromarray(rgb), frame_score(rgb)))
            if len(frames) >= samples:
                break
        return frames
    finally:
        reader.close()


def pick_candidates(frames: list[Frame], count: int = CANDIDATES) -> list[Frame]:
    best = sorted(frames, key=lambda f: f.score, reverse=True)[:count]
    return sorted(best, key=lambda f: f.index)


def contact_sheet(frames: list[Frame]) -> Image.Image:
    cols = min(2, len(frames))
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * TILE, rows * TILE), "black")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=TILE // 8)
    for number, frame in enumerate(frames, start=1):
        tile = frame.image.copy()
        tile.thumbnail((TILE, TILE))
        x = ((number - 1) % cols) * TILE
        y = ((number - 1) // cols) * TILE
        sheet.paste(tile, (x + (TILE - tile.width) // 2, y + (TILE - tile.height) // 2))
        draw.text((x + 10, y + 6), str(number), font=font, fill="yellow", stroke_width=3,
                  stroke_fill="black")
    return sheet
