import base64
import io

from PIL import Image

Image.MAX_IMAGE_PIXELS = 40_000_000

PREVIEW_SIDE = 1024


class BadImageError(ValueError):
    pass


def load_image(data: bytes) -> Image.Image:
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise BadImageError("file is not a readable image") from exc
    return img.convert("RGB")


def to_preview_b64(img: Image.Image, side: int = PREVIEW_SIDE) -> str:
    preview = img.copy()
    preview.thumbnail((side, side))
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
