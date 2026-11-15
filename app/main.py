import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from langchain_core.language_models import BaseChatModel

from app.chains import build_caption_chain, build_edit_chain, build_thumbnail_chain, make_llm
from app.imaging import BadImageError, load_image, to_png_b64, to_preview_b64
from app.ops import apply_plan
from app.schemas import CaptionResult, EditPlan, ThumbnailChoice
from app.video import contact_sheet, pick_candidates, sample_frames

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 60 * 1024 * 1024
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Media Chain Studio")


@lru_cache
def get_llm() -> BaseChatModel:
    return make_llm()


Llm = Annotated[BaseChatModel, Depends(get_llm)]


async def read_limited(upload: UploadFile, limit: int) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"file is larger than {limit // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "empty file")
    return data


async def read_image(upload: UploadFile):
    data = await read_limited(upload, MAX_IMAGE_BYTES)
    try:
        return load_image(data)
    except BadImageError as exc:
        raise HTTPException(415, str(exc)) from exc


def run_chain(chain, inputs: dict):
    try:
        return chain.invoke(inputs)
    except Exception as exc:
        raise HTTPException(502, f"model call failed: {type(exc).__name__}") from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/edit")
async def edit(
    llm: Llm,
    image: Annotated[UploadFile, File()],
    instruction: Annotated[str, Form(min_length=3, max_length=500)],
) -> dict:
    img = await read_image(image)
    plan: EditPlan = run_chain(
        build_edit_chain(llm), {"image_b64": to_preview_b64(img), "instruction": instruction}
    )
    result, warnings = apply_plan(img, plan)
    return {
        "plan": plan.model_dump(),
        "warnings": warnings,
        "size": list(result.size),
        "image_png_b64": to_png_b64(result),
    }


@app.post("/api/caption")
async def caption(
    llm: Llm,
    image: Annotated[UploadFile, File()],
    language: Annotated[str, Form(max_length=30)] = "English",
) -> CaptionResult:
    img = await read_image(image)
    return run_chain(
        build_caption_chain(llm), {"image_b64": to_preview_b64(img), "language": language}
    )


@app.post("/api/video/thumbnail")
async def thumbnail(llm: Llm, video: Annotated[UploadFile, File()]) -> dict:
    data = await read_limited(video, MAX_VIDEO_BYTES)
    suffix = Path(video.filename or "clip.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".webm", ".mkv"}:
        raise HTTPException(415, "unsupported video format")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        try:
            frames = sample_frames(path)
        except Exception as exc:
            raise HTTPException(415, "could not decode video") from exc
    if not frames:
        raise HTTPException(415, "video has no frames")
    candidates = pick_candidates(frames)
    sheet = contact_sheet(candidates)
    choice: ThumbnailChoice = run_chain(
        build_thumbnail_chain(llm),
        {"sheet_b64": to_preview_b64(sheet, side=1024), "count": len(candidates)},
    )
    position = min(max(choice.best_index, 1), len(candidates)) - 1
    chosen = candidates[position]
    return {
        "choice": choice.model_dump(),
        "frame_index": chosen.index,
        "candidates": [{"frame_index": c.index, "score": round(c.score, 1)} for c in candidates],
        "image_png_b64": to_png_b64(chosen.image),
    }
