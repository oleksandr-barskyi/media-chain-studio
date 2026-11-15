import imageio.v2 as imageio
import numpy as np
import pytest

from app.video import contact_sheet, frame_score, pick_candidates, sample_frames


def checkerboard(size=64, cell=8):
    y, x = np.indices((size, size))
    board = (((x // cell) + (y // cell)) % 2 * 255).astype(np.uint8)
    return np.stack([board] * 3, axis=2)


def test_sharp_frame_beats_flat_and_dark():
    flat = np.full((64, 64, 3), 128, np.uint8)
    dark = (checkerboard() * 0.05).astype(np.uint8)
    assert frame_score(checkerboard()) > frame_score(flat)
    assert frame_score(checkerboard()) > frame_score(dark)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    frames = []
    for i in range(48):
        if 20 <= i < 28:
            frames.append(checkerboard())
        else:
            frames.append(np.full((64, 64, 3), 10 + i, np.uint8))
    imageio.mimsave(str(path), frames, fps=12, macro_block_size=16)
    return path


def test_sampling_and_candidates_prefer_detailed_frames(clip):
    frames = sample_frames(clip, samples=16)
    assert 8 <= len(frames) <= 16
    best = pick_candidates(frames, count=2)
    assert all(20 <= f.index < 28 for f in best)


def test_contact_sheet_has_one_tile_per_candidate(clip):
    frames = pick_candidates(sample_frames(clip), count=4)
    sheet = contact_sheet(frames)
    assert sheet.size == (768, 768)
