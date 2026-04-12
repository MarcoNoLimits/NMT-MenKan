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

**Important:** On a slim branch used only for Spaces, do **not** run `git add .` from the repo root. That can stage logs, `Report/*.pdf`, and the C++ tree and will make Hugging Face reject the push. Stage files explicitly (for example `git add .dockerignore README.md Dockerfile requirements.txt scripts/ nllb_int8/ .gitattributes`) or rely on `.gitignore` patterns below.

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

### Production client (recommended)

Use `scripts/nmt_client.py` from your app or jobs so every call uses the correct JSON and headers:

```python
from scripts.nmt_client import translate

r = translate("https://YOUR_USERNAME-YOUR_SPACE.hf.space", "Hello, how are you?", api_key="YOUR_KEY")
print(r.translation)
```

### Request formats (all supported on `POST /translate`)

- **JSON:** `Content-Type: application/json`, body `{"text":"..."}` (preferred for production)
- **Form:** `application/x-www-form-urlencoded` or `multipart/form-data` with field `text`
- **Plain:** `Content-Type: text/plain`, body is the sentence only
- **GET (convenience):** `GET /translate?text=...` (URLs are length-limited; prefer POST for long text)

### Validation errors

Malformed JSON or missing `text` returns **400** with an `hint` field. Typos in JSON shape that reach Pydantic return **422** with the same `hint`.

## 5) Free-tier expectations

- The Space can sleep when idle.
- First request after sleep can be slow (cold start).
- CPU/RAM limits may constrain throughput for large bursts.

For better uptime/performance, move to paid hardware or Cloud Run later.
