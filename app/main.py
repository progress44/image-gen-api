"""FastAPI app for OpenAI-style image generation."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import Config
from app.model import image_model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_task = asyncio.create_task(asyncio.to_thread(image_model.initialize))
    try:
        yield
    finally:
        if not init_task.done():
            init_task.cancel()
            with suppress(asyncio.CancelledError):
                await init_task


def _docs_url(path: str) -> str | None:
    return path if Config.ENABLE_DOCS else None


app = FastAPI(
    title="Image Generation API",
    description="OpenAI-compatible image generation API",
    version="1.2.0",
    docs_url=_docs_url("/docs"),
    redoc_url=_docs_url("/redoc"),
    lifespan=lifespan,
)

cors_origins = Config.CORS_ORIGINS
allowed_origins = ["*"] if cors_origins == "*" else [o.strip() for o in cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageGenerationRequest(BaseModel):
    model: str = Field(default=Config.MODEL_ID)
    prompt: str
    n: int = Field(default=1, ge=1)
    size: str = Field(default="1024x1024")
    response_format: str = Field(default="url")
    negative_prompt: str | None = None
    num_inference_steps: int | None = Field(default=None, ge=1)
    guidance_scale: float | None = Field(default=None, ge=0.0)
    seed: int | None = Field(default=None, ge=0)


_SIZE_PATTERN = re.compile(r"^(\d+)x(\d+)$")


def _parse_size(size: str) -> tuple[int, int]:
    match = _SIZE_PATTERN.fullmatch(size.strip())
    if not match:
        raise HTTPException(status_code=400, detail={"error": {"message": "size must be WIDTHxHEIGHT"}})

    width = int(match.group(1))
    height = int(match.group(2))

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail={"error": {"message": "size must be positive"}})

    if width > Config.MAX_WIDTH or height > Config.MAX_HEIGHT:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"Requested size {width}x{height} exceeds maximum "
                        f"{Config.MAX_WIDTH}x{Config.MAX_HEIGHT}"
                    )
                }
            },
        )

    return width, height


def _normalize_response_format(response_format: str) -> str:
    rf = response_format.strip().lower()
    if rf not in {"url", "b64_json"}:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "response_format must be one of: url, b64_json"}},
        )
    return rf


def _save_image_and_get_path(image_bytes: bytes, extension: str) -> tuple[str, str]:
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    image_id = f"{uuid.uuid4().hex}.{extension}"
    final_path = os.path.join(Config.OUTPUT_DIR, image_id)

    with tempfile.NamedTemporaryFile(
        dir=Config.OUTPUT_DIR,
        prefix="img-",
        suffix=f".{extension}",
        delete=False,
    ) as tmp:
        tmp.write(image_bytes)
        temp_path = tmp.name

    os.replace(temp_path, final_path)
    return image_id, final_path


@app.get("/")
def root() -> dict[str, object]:
    status_data = image_model.status()
    return {
        "name": "Image Generation API",
        "model": status_data.model_id,
        "device": status_data.device,
        "initialized": status_data.initialized,
        "docs": "/docs" if Config.ENABLE_DOCS else None,
    }


@app.get("/health")
def health() -> dict[str, object]:
    status_data = image_model.status()
    return {
        "status": "ok",
        "model": status_data.model_id,
        "initialized": status_data.initialized,
        "initializing": status_data.initializing,
        "device": status_data.device,
        "error": status_data.error,
    }


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": Config.MODEL_ID, "object": "model"}]}


@app.get("/v1/images/{image_id}")
def get_image(image_id: str):
    safe_name = os.path.basename(image_id)
    path = os.path.join(Config.OUTPUT_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail={"error": {"message": "image not found"}})
    return FileResponse(path)


@app.post("/v1/images/generations")
async def generate_image(payload: ImageGenerationRequest, request: Request):
    if payload.model != Config.MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"Only model '{Config.MODEL_ID}' is available."}},
        )

    if payload.n != 1:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "Only n=1 is currently supported."}},
        )

    width, height = _parse_size(payload.size)
    response_format = _normalize_response_format(payload.response_format)
    num_inference_steps = payload.num_inference_steps or Config.DEFAULT_STEPS
    guidance_scale = payload.guidance_scale if payload.guidance_scale is not None else Config.DEFAULT_GUIDANCE_SCALE

    if num_inference_steps > Config.MAX_STEPS:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"num_inference_steps cannot exceed {Config.MAX_STEPS}"}},
        )

    image = await asyncio.to_thread(
        image_model.generate,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=payload.seed,
    )

    image_format = "JPEG" if Config.OUTPUT_FORMAT in {"jpeg", "jpg"} else "PNG"
    extension = "jpg" if image_format == "JPEG" else "png"

    buffer = io.BytesIO()
    save_kwargs = {"quality": 95} if image_format == "JPEG" else {}
    image.save(buffer, format=image_format, **save_kwargs)
    image_bytes = buffer.getvalue()

    if response_format == "b64_json":
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        return {"created": int(time.time()), "data": [{"b64_json": b64_data}]}

    image_id, _path = _save_image_and_get_path(image_bytes, extension)
    base_url = str(request.base_url).rstrip("/")
    return {"created": int(time.time()), "data": [{"url": f"{base_url}/v1/images/{image_id}"}]}

