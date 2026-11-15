from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from app.schemas import EditPlan, EditStep

MAX_STEPS = 8
MAX_SIDE = 4096
ASPECTS = {"1:1": 1.0, "4:5": 4 / 5, "16:9": 16 / 9, "9:16": 9 / 16, "3:2": 3 / 2}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def crop_to_aspect(img: Image.Image, aspect: str) -> Image.Image:
    target = ASPECTS[aspect]
    width, height = img.size
    if width / height > target:
        new_width = round(height * target)
        left = (width - new_width) // 2
        return img.crop((left, 0, left + new_width, height))
    new_height = round(width / target)
    top = (height - new_height) // 2
    return img.crop((0, top, width, top + new_height))


def resize_longest(img: Image.Image, side: float) -> Image.Image:
    side = int(clamp(side, 16, MAX_SIDE))
    width, height = img.size
    scale = side / max(width, height)
    return img.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)


def sepia(img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(img)
    return ImageOps.colorize(gray, black="#2b1d0e", white="#f5e6c8")


def add_text(img: Image.Image, text: str, position: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    size = max(12, out.width // 18)
    font = ImageFont.load_default(size=size)
    text = text[:80]
    box = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = box[2] - box[0], box[3] - box[1]
    x = (out.width - text_width) // 2
    margin = out.height // 20
    y = {
        "top": margin,
        "center": (out.height - text_height) // 2,
        "bottom": out.height - text_height - margin,
    }[position]
    stroke = max(1, size // 12)
    draw.text((x, y), text, font=font, fill="white", stroke_width=stroke, stroke_fill="black")
    return out


def apply_step(img: Image.Image, step: EditStep) -> tuple[Image.Image, str | None]:
    op = step.op
    if op == "crop_aspect":
        if step.aspect is None:
            return img, "crop_aspect without aspect skipped"
        return crop_to_aspect(img, step.aspect), None
    if op == "resize":
        if step.value is None:
            return img, "resize without size skipped"
        return resize_longest(img, step.value), None
    if op == "rotate":
        return img.rotate(clamp(step.value or 0, -360, 360), expand=True), None
    if op == "flip":
        if step.direction == "vertical":
            return ImageOps.flip(img), None
        return ImageOps.mirror(img), None
    if op in ("brightness", "contrast", "saturation"):
        enhancer = {
            "brightness": ImageEnhance.Brightness,
            "contrast": ImageEnhance.Contrast,
            "saturation": ImageEnhance.Color,
        }[op]
        return enhancer(img).enhance(clamp(step.value or 1.0, 0.0, 3.0)), None
    if op == "blur":
        return img.filter(ImageFilter.GaussianBlur(clamp(step.value or 2.0, 0.0, 25.0))), None
    if op == "sharpen":
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)), None
    if op == "grayscale":
        return ImageOps.grayscale(img).convert("RGB"), None
    if op == "sepia":
        return sepia(img), None
    if op == "add_text":
        if not step.text:
            return img, "add_text without text skipped"
        return add_text(img, step.text, step.position or "bottom"), None
    return img, f"unknown operation {op} skipped"


def apply_plan(img: Image.Image, plan: EditPlan) -> tuple[Image.Image, list[str]]:
    out = img.convert("RGB")
    warnings: list[str] = []
    steps = plan.steps
    if len(steps) > MAX_STEPS:
        warnings.append(f"plan had {len(steps)} steps, applied first {MAX_STEPS}")
        steps = steps[:MAX_STEPS]
    for step in steps:
        out, warning = apply_step(out, step)
        if warning:
            warnings.append(warning)
    return out, warnings
