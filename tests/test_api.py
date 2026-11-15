import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_llm
from app.schemas import CaptionResult, EditPlan, EditStep, ThumbnailChoice


def step(op, **kw) -> EditStep:
    fields = {"value": None, "aspect": None, "direction": None, "text": None, "position": None}
    fields.update(kw)
    return EditStep(op=op, **fields)


@pytest.fixture
def client_with(fake_model):
    def build(outputs):
        model = fake_model(outputs)
        app.dependency_overrides[get_llm] = lambda: model
        return TestClient(app), model

    yield build
    app.dependency_overrides.clear()


def test_edit_applies_plan_from_model(client_with, sample_png):
    plan = EditPlan(
        steps=[step("crop_aspect", aspect="1:1"), step("add_text", text="Hi", position="top")],
        summary="Square with a title",
        unsupported=None,
    )
    client, _ = client_with({EditPlan: plan})
    res = client.post(
        "/api/edit",
        files={"image": ("a.png", sample_png, "image/png")},
        data={"instruction": "square with a title"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["size"] == [300, 300]
    out = Image.open(io.BytesIO(base64.b64decode(body["image_png_b64"])))
    assert out.size == (300, 300)


def test_edit_rejects_non_image(client_with):
    client, model = client_with({})
    res = client.post(
        "/api/edit",
        files={"image": ("a.png", b"not an image", "image/png")},
        data={"instruction": "make it nice"},
    )
    assert res.status_code == 415
    assert model.calls == []


def test_model_failure_becomes_502(client_with, sample_png):
    client, _ = client_with({CaptionResult: RuntimeError("boom")})
    res = client.post("/api/caption", files={"image": ("a.png", sample_png, "image/png")})
    assert res.status_code == 502


def test_caption_returns_structured_result(client_with, sample_png):
    caption = CaptionResult(alt_text="Yellow square", title="Sun", hashtags=["art"], mood="warm")
    client, _ = client_with({CaptionResult: caption})
    res = client.post("/api/caption", files={"image": ("a.png", sample_png, "image/png")})
    assert res.status_code == 200
    assert res.json()["title"] == "Sun"


def test_thumbnail_clamps_out_of_range_choice(client_with, tmp_path):
    import imageio.v2 as imageio
    import numpy as np

    path = tmp_path / "c.mp4"
    imageio.mimsave(str(path), [np.full((32, 32, 3), 40 + i * 5, np.uint8) for i in range(24)],
                    fps=12, macro_block_size=16)
    choice = ThumbnailChoice(best_index=99, reason="sharp", title="Clip")
    client, _ = client_with({ThumbnailChoice: choice})
    res = client.post(
        "/api/video/thumbnail", files={"video": ("c.mp4", path.read_bytes(), "video/mp4")}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["frame_index"] == body["candidates"][-1]["frame_index"]


def test_unsupported_video_extension(client_with):
    client, _ = client_with({})
    res = client.post("/api/video/thumbnail", files={"video": ("a.exe", b"xx", "video/mp4")})
    assert res.status_code == 415
