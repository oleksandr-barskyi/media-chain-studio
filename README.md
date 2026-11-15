# Media Chain Studio

Image and video features in the spirit of consumer photo apps, built with **LangChain** and
**Claude** (vision + structured outputs) on a **FastAPI** backend:

- **Edit by prompt.** "Square crop for Instagram, a bit warmer, add 'Summer sale' at the
  bottom" becomes a typed edit plan, and Pillow applies it. The model plans, the code edits.
- **Caption and alt text.** Accessible description, short title, hashtags and mood, in any
  language.
- **Video thumbnail.** Frames are sampled with ffmpeg, scored for sharpness and exposure, the
  best candidates go to the model as one numbered contact sheet, and the model picks the
  thumbnail and suggests a title.

A one-page UI is served at `/`.

## Why it is built this way

| Decision | Why |
| --- | --- |
| The model returns a plan, never pixels or code | Every operation is a known function with clamped parameters, so a bad answer cannot break the image or the server |
| `with_structured_output(..., method="json_schema")` | Claude's structured outputs validate against the Pydantic schema. Claude Opus 5.5 rejects forced tool calls, so the function-calling method is not used |
| Images are downscaled to 1024 px JPEG before the call | The plan does not need full resolution; editing still runs on the original |
| Video: cheap local scoring first, the model only judges 4 frames on one sheet | One image call instead of one per frame, and dark or blurry transition frames never reach the model |
| User text is fenced in `<request>` tags and treated as data | Instructions hidden in a request cannot change the system prompt |
| Server-side fallback on refusal (`fallbacks: "default"`) | A refused request is retried on a fallback model by the API instead of failing |
| Model behind a FastAPI dependency | Tests swap in a fake model; nothing in the test suite touches the network |

## Run

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"     # .venv/bin/pip on macOS and Linux
cp .env.example .env                       # set ANTHROPIC_API_KEY
.venv/Scripts/uvicorn app.main:app --reload
```

Open http://localhost:8000.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | | API key |
| `STUDIO_MODEL` | `claude-opus-5-5` | Model id |
| `STUDIO_EFFORT` | `low` | Effort level for the planning calls |
| `STUDIO_FALLBACKS` | `on` | Server-side fallback when a request is refused |

## API

| Endpoint | Input | Output |
| --- | --- | --- |
| `POST /api/edit` | `image`, `instruction` | plan, warnings, edited PNG (base64) |
| `POST /api/caption` | `image`, optional `language` | alt text, title, hashtags, mood |
| `POST /api/video/thumbnail` | `video` (mp4, mov, webm, mkv; up to 60 MB) | choice, candidate frames, thumbnail PNG (base64) |

Edit operations: crop to aspect (1:1, 4:5, 16:9, 9:16, 3:2), resize, rotate, flip, brightness,
contrast, saturation, blur, sharpen, grayscale, sepia, text overlay.

## Tests

```bash
.venv/Scripts/ruff check .
.venv/Scripts/pytest -q
```

22 tests: image operations and clamping, chain inputs (image block, fenced and truncated
request, language), video sampling and candidate ranking on a generated clip, and the API
end to end with a fake model (bad uploads, model failure, out-of-range answers). CI runs the
same on every push.

## Layout

```
app/
  schemas.py   Pydantic schemas the model must fill: EditPlan, CaptionResult, ThumbnailChoice
  chains.py    LangChain chains: prompt -> ChatAnthropic -> structured output
  ops.py       Pillow operations and plan execution with clamped parameters
  video.py     frame sampling, sharpness/exposure score, contact sheet
  imaging.py   safe image loading, previews, encoding
  main.py      FastAPI endpoints and upload limits
  static/      one-page UI
tests/
```
