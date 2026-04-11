# Hugging Face Spaces Deployment (Free-Friendly)

This repo is ready for a Docker Space that serves:

- `GET /healthz`
- `POST /translate`

from `scripts/nmt_http_api.py`.

## 1) Create a Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Choose:
   - **SDK**: `Docker`
   - **Visibility**: Public (or Private)
4. Create the Space.

## 2) Push this repo to the Space

Use the Space Git URL as your remote and push:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push hf main
```

If your default branch is not `main`, push that branch.

## 3) Configure secrets/variables

In Space **Settings**:

- Set `REQUIRE_API_KEY=1` (recommended for public usage)
- Set `NMT_API_KEY=<long-random-string>`

Optional vars:

- `MAX_INPUT_CHARS=2000`
- `TRANSLATION_TIMEOUT_MS=10000`

If you leave `REQUIRE_API_KEY=0`, endpoint stays open.

## 4) Run and verify

Once the build is green, test:

Health:

```bash
curl https://YOUR_SPACE_SUBDOMAIN.hf.space/healthz
```

Translate:

```bash
curl -X POST https://YOUR_SPACE_SUBDOMAIN.hf.space/translate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"text":"Hello, how are you?"}'
```

## 5) Free-tier expectations

- The Space can sleep when idle.
- First request after sleep can be slow (cold start).
- CPU/RAM limits may constrain throughput for large bursts.

For better uptime/performance, move to paid hardware or Cloud Run later.
