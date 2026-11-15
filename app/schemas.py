from typing import Literal

from pydantic import BaseModel, Field

OpName = Literal[
    "crop_aspect",
    "resize",
    "rotate",
    "flip",
    "brightness",
    "contrast",
    "saturation",
    "blur",
    "sharpen",
    "grayscale",
    "sepia",
    "add_text",
]


class EditStep(BaseModel):
    op: OpName = Field(description="Operation to apply, in order.")
    value: float | None = Field(
        description=(
            "Numeric parameter. brightness/contrast/saturation: factor, 1.0 = unchanged. "
            "rotate: degrees counterclockwise. blur: radius in px. resize: longest side in px. "
            "Null for operations without a number."
        )
    )
    aspect: Literal["1:1", "4:5", "16:9", "9:16", "3:2"] | None = Field(
        description="Target aspect for crop_aspect, null otherwise."
    )
    direction: Literal["horizontal", "vertical"] | None = Field(
        description="Direction for flip, null otherwise."
    )
    text: str | None = Field(description="Text for add_text, null otherwise.")
    position: Literal["top", "center", "bottom"] | None = Field(
        description="Placement for add_text, null otherwise."
    )


class EditPlan(BaseModel):
    steps: list[EditStep] = Field(description="At most 8 steps that fulfil the instruction.")
    summary: str = Field(description="One sentence for the user describing the edit.")
    unsupported: str | None = Field(
        description="What part of the request cannot be done with these operations, or null."
    )


class CaptionResult(BaseModel):
    alt_text: str = Field(description="Accessible description of the image, at most 2 sentences.")
    title: str = Field(description="Short catchy title for a social post, at most 8 words.")
    hashtags: list[str] = Field(description="3 to 6 relevant hashtags without the # sign.")
    mood: str = Field(description="One or two words for the overall mood.")


class ThumbnailChoice(BaseModel):
    best_index: int = Field(description="Number of the tile chosen as the video thumbnail.")
    reason: str = Field(description="One sentence why this frame works as a thumbnail.")
    title: str = Field(description="Suggested video title, at most 10 words.")
