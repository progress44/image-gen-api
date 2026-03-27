"""Configuration for Image Generation API."""

from __future__ import annotations

import os


class Config:
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8001"))

    MODEL_ID = os.getenv("IMAGE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
    DEVICE = os.getenv("IMAGE_DEVICE", "auto")
    TORCH_DTYPE = os.getenv("IMAGE_TORCH_DTYPE", "auto")

    DEFAULT_STEPS = int(os.getenv("IMAGE_DEFAULT_STEPS", "30"))
    MAX_STEPS = int(os.getenv("IMAGE_MAX_STEPS", "80"))
    DEFAULT_GUIDANCE_SCALE = float(os.getenv("IMAGE_DEFAULT_GUIDANCE_SCALE", "7.5"))

    MAX_WIDTH = int(os.getenv("IMAGE_MAX_WIDTH", "1536"))
    MAX_HEIGHT = int(os.getenv("IMAGE_MAX_HEIGHT", "1536"))

    OUTPUT_DIR = os.getenv("IMAGE_OUTPUT_DIR", "/data/outputs")
    OUTPUT_FORMAT = os.getenv("IMAGE_OUTPUT_FORMAT", "png").lower()

    ENABLE_DOCS = os.getenv("IMAGE_ENABLE_DOCS", "true").lower() == "true"
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    @classmethod
    def validate(cls) -> None:
        if cls.PORT <= 0:
            raise ValueError("PORT must be positive")
        if cls.DEFAULT_STEPS <= 0:
            raise ValueError("IMAGE_DEFAULT_STEPS must be positive")
        if cls.MAX_STEPS <= 0:
            raise ValueError("IMAGE_MAX_STEPS must be positive")
        if cls.DEFAULT_STEPS > cls.MAX_STEPS:
            raise ValueError("IMAGE_DEFAULT_STEPS cannot exceed IMAGE_MAX_STEPS")
        if cls.MAX_WIDTH <= 0 or cls.MAX_HEIGHT <= 0:
            raise ValueError("IMAGE_MAX_WIDTH and IMAGE_MAX_HEIGHT must be positive")
        if cls.OUTPUT_FORMAT not in {"png", "jpeg", "jpg"}:
            raise ValueError("IMAGE_OUTPUT_FORMAT must be png, jpeg, or jpg")
        if not cls.MODEL_ID.strip():
            raise ValueError("IMAGE_MODEL_ID cannot be empty")
