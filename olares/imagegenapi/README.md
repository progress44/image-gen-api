# Image Generation API for Olares

This package deploys the published image:

- `ghcr.io/progress44/rpi-system-image-gen-api:latest`

The app exposes OpenAI-style image generation endpoints at:

- `http://imagegenapi-svc:8001`

## Endpoints

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/images/generations`
- `GET /v1/images/{image_id}`

## Request example

```bash
curl -X POST http://imagegenapi-svc:8001/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "a cinematic mountain landscape at sunrise",
    "size": "1024x1024",
    "response_format": "url"
  }'
```

## Notes

- The first request may be slower while model files are downloaded.
- Hugging Face and torch caches persist under `userspace.appData`.
- Generated outputs persist under `userspace.appData/outputs`.
- Use Olares env variables `OLARES_USER_HUGGINGFACE_TOKEN` and
  `OLARES_USER_HUGGINGFACE_SERVICE` if needed.
