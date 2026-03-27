# Image Generation API

FastAPI wrapper that exposes OpenAI-style image generation endpoints backed by
Diffusers.

## Endpoints

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/images/generations` (OpenAI-compatible JSON payload)
- `GET /v1/images/{image_id}` (serves generated images when using `response_format=url`)

## OpenAI-style request example

```bash
curl -X POST http://localhost:8001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "a cinematic mountain landscape at sunrise",
    "size": "1024x1024",
    "response_format": "url"
  }'
```

## Environment Variables

- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8001`)
- `IMAGE_MODEL_ID` (default: `stabilityai/stable-diffusion-xl-base-1.0`)
- `IMAGE_DEVICE` (`auto|cpu|cuda|mps`, default: `auto`)
- `IMAGE_TORCH_DTYPE` (`auto|float16|float32|bfloat16`, default: `auto`)
- `IMAGE_ENABLE_DOCS` (`true|false`, default: `true`)
- `IMAGE_DEFAULT_STEPS` (default: `30`)
- `IMAGE_MAX_STEPS` (default: `80`)
- `IMAGE_MAX_WIDTH` (default: `1536`)
- `IMAGE_MAX_HEIGHT` (default: `1536`)
- `IMAGE_OUTPUT_DIR` (default: `/data/outputs`)
- `IMAGE_OUTPUT_FORMAT` (`png|jpeg|jpg`, default: `png`)

