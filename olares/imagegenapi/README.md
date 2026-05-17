# Imogen Image Generation API for Olares

This package deploys Imogen as an Olares shared application using the published image:

- `ghcr.io/progress44/rpi-system-image-gen-api:latest`

The administrator installs one shared GPU-backed backend for the Olares cluster.
Each user gets a lightweight user-space API entrance that proxies to that shared backend.

## Olares Endpoints

- Shared app-to-app endpoint: `http://imagegenapi.shared.olares.com`
- User-space endpoint: `https://imagegenapi.{OlaresID}.olares.com`

Use the shared endpoint for backend-to-backend calls from other Olares apps. Use the
user-space endpoint when a browser or user-installed app needs the normal Olares route.

## Chart Structure

- `imagegenapiserver`: shared backend chart deployed once for the cluster by the admin.
- `imagegenapi`: per-user OpenResty proxy chart that exposes the normal user-space API entrance.

## Endpoints

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/images/generations`
- `GET /v1/images/{image_id}`

## Request example

```bash
curl -X POST http://imagegenapi.shared.olares.com/v1/images/generations \
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
- Hugging Face and torch caches persist with the shared backend installation.
- Generated outputs persist with the shared backend under `outputs`.
- Use Olares env variables `OLARES_USER_HUGGINGFACE_TOKEN` and
  `OLARES_USER_HUGGINGFACE_SERVICE` on the admin install if needed.
