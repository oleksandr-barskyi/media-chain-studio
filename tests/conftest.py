import io

import numpy as np
import pytest
from langchain_core.runnables import RunnableLambda
from PIL import Image


class FakeStructuredModel:
    def __init__(self, outputs: dict):
        self.outputs = outputs
        self.calls: list = []

    def with_structured_output(self, schema, method="function_calling", **kwargs):
        assert method == "json_schema"

        def respond(messages):
            self.calls.append((schema, messages))
            result = self.outputs[schema]
            if isinstance(result, Exception):
                raise result
            return result

        return RunnableLambda(respond)


@pytest.fixture
def fake_model():
    return FakeStructuredModel


def make_image(width=400, height=300, color=(120, 160, 200)) -> Image.Image:
    img = Image.new("RGB", (width, height), color)
    pixels = np.array(img)
    pixels[: height // 2, : width // 2] = (250, 240, 30)
    return Image.fromarray(pixels)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_png() -> bytes:
    return png_bytes(make_image())
