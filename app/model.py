"""Model lifecycle and image generation logic."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

from app.config import Config


@dataclass
class ModelStatus:
    initialized: bool
    initializing: bool
    model_id: str
    device: str
    error: str | None


class ImageModel:
    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._initializing = False
        self._error: str | None = None
        self._resolved_device = self._resolve_device()

    def _resolve_device(self) -> str:
        configured = Config.DEVICE.lower()
        if configured != "auto":
            return configured

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _resolve_torch_dtype(self) -> torch.dtype:
        configured = Config.TORCH_DTYPE.lower()

        if configured == "float16":
            return torch.float16
        if configured == "bfloat16":
            return torch.bfloat16
        if configured == "float32":
            return torch.float32

        return torch.float16 if self._resolved_device == "cuda" else torch.float32

    def initialize(self) -> None:
        if self._pipe is not None:
            return

        with self._lock:
            if self._pipe is not None:
                return

            self._initializing = True
            self._error = None
            try:
                self._pipe = AutoPipelineForText2Image.from_pretrained(
                    Config.MODEL_ID,
                    torch_dtype=self._resolve_torch_dtype(),
                    use_safetensors=True,
                )
                self._pipe = self._pipe.to(self._resolved_device)
                if self._resolved_device == "cuda":
                    self._pipe.enable_attention_slicing()
                os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                raise
            finally:
                self._initializing = False

    def status(self) -> ModelStatus:
        return ModelStatus(
            initialized=self._pipe is not None,
            initializing=self._initializing,
            model_id=Config.MODEL_ID,
            device=self._resolved_device,
            error=self._error,
        )

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> Image.Image:
        self.initialize()

        if self._pipe is None:
            raise RuntimeError("Image pipeline is not initialized")

        generator = None
        if seed is not None:
            device_for_seed = self._resolved_device if self._resolved_device == "cuda" else "cpu"
            generator = torch.Generator(device=device_for_seed).manual_seed(seed)

        with self._generate_lock:
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        if not result.images:
            raise RuntimeError("Model returned no images")

        return result.images[0]


image_model = ImageModel()
