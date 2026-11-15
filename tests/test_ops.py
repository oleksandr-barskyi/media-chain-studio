import pytest

from app.ops import MAX_STEPS, apply_plan, crop_to_aspect, resize_longest
from app.schemas import EditPlan, EditStep
from tests.conftest import make_image


def step(op, **kw) -> EditStep:
    fields = {"value": None, "aspect": None, "direction": None, "text": None, "position": None}
    fields.update(kw)
    return EditStep(op=op, **fields)


def plan(*steps) -> EditPlan:
    return EditPlan(steps=list(steps), summary="test", unsupported=None)


@pytest.mark.parametrize(
    ("aspect", "expected"),
    [("1:1", (300, 300)), ("16:9", (400, 225)), ("9:16", (169, 300)), ("4:5", (240, 300))],
)
def test_crop_to_aspect_keeps_center(aspect, expected):
    assert crop_to_aspect(make_image(400, 300), aspect).size == expected


def test_resize_longest_is_clamped():
    assert max(resize_longest(make_image(400, 300), 100_000).size) == 4096
    assert resize_longest(make_image(400, 300), 200).size == (200, 150)


def test_brightness_factor_is_clamped_to_safe_range():
    img = make_image(color=(100, 100, 100))
    out, warnings = apply_plan(img, plan(step("brightness", value=50)))
    assert warnings == []
    assert out.getpixel((399, 299)) == (255, 255, 255)


def test_grayscale_then_text_runs_in_order():
    out, warnings = apply_plan(
        make_image(), plan(step("grayscale"), step("add_text", text="Sale", position="bottom"))
    )
    r, g, b = out.getpixel((10, 10))
    assert r == g == b
    assert warnings == []


def test_steps_missing_parameters_are_skipped_with_warning():
    out, warnings = apply_plan(make_image(), plan(step("crop_aspect"), step("add_text")))
    assert out.size == (400, 300)
    assert len(warnings) == 2


def test_long_plans_are_truncated():
    steps = [step("sharpen") for _ in range(MAX_STEPS + 3)]
    _, warnings = apply_plan(make_image(), plan(*steps))
    assert any("applied first" in w for w in warnings)


def test_rotate_expands_canvas():
    out, _ = apply_plan(make_image(400, 300), plan(step("rotate", value=90)))
    assert out.size == (300, 400)
