# Hugging Face Spaces — NMT MenKan (this project)

**Space (Hub page):** [https://huggingface.co/spaces/marconolimits/NMT](https://huggingface.co/spaces/marconolimits/NMT)

**Public app base URL (API):** `https://marconolimits-nmt.hf.space`

**Git remote (HTTPS):** `https://huggingface.co/spaces/marconolimits/NMT`

Endpoints (from `scripts/nmt_http_api.py`):

- `GET /` — short HTML help
- `GET /healthz` — health check
- `GET /translate?text=...` — translate (short text; prefer POST for long text)
- `POST /translate` — translate (JSON, form, or plain text body)

---

## 1) Create a Space (already done for you)

If you recreate elsewhere: [Spaces](https://huggingface.co/spaces) → **Create new Space** → SDK **Docker**.

---

## 2) Push this repo to the Space

Add the remote once (skip if `hf` already exists):

```bash
git remote add hf https://huggingface.co/spaces/marconolimits/NMT
```

This branch is deployed with:

```bash
git push hf hf-space:main
```

If your deploy branch is named differently, replace `hf-space`.

**Do not** run `git add .` from the repo root on the deploy branch. Stage only what the Space needs (for example `Dockerfile`, `requirements.txt`, `scripts/`, `nllb_int8/`, `README.md`, `.dockerignore`, `.gitattributes`, `.gitignore`) or rely on `.gitignore` so Hub rejects no binary junk.

---

## 3) API key (production)

The server reads **`NMT_API_KEY`** (secret) and **`REQUIRE_API_KEY`** (`1` = enforce).  
Clients must send **`X-API-Key: <same value>`** on **`POST /translate`**, **`GET /translate`**, and **`GET /`** does not need a key. **`GET /healthz`** stays public for probes.

### 3a) On Hugging Face (this Space)

1. Open **[Space → Settings → Variables and secrets](https://huggingface.co/spaces/marconolimits/NMT/settings)**.
2. Under **Variables**, add:
   - Name: `REQUIRE_API_KEY` → Value: `1`
3. Under **Secrets** (or **Repository secrets**), add:
   - Name: `NMT_API_KEY` → Value: a long random string (example generator below).  
   - Treat this like a password; save it in your password manager for clients.

4. **Factory reboot:** open the Space **App** tab or use **Restart** so the container picks up env vars.

5. Test (replace `PASTE_YOUR_KEY_HERE`):

```bash
curl -sS "https://marconolimits-nmt.hf.space/healthz"
curl -sS -X POST "https://marconolimits-nmt.hf.space/translate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: PASTE_YOUR_KEY_HERE" \
  -d '{"text":"Hello"}'
```

A wrong or missing key returns **`401`** with `{"detail":"Unauthorized"}`.

### 3b) Local development

Copy **`.env.example`** to **`.env`**, set `NMT_API_KEY` and `REQUIRE_API_KEY=1`, then run uvicorn from the repo root. **`.env` is gitignored** — never commit it.

Generate a key (any one of these):

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Optional variables (HF or `.env`)

| Name | Example |
|------|---------|
| `MAX_INPUT_CHARS` | `2000` |
| `TRANSLATION_TIMEOUT_MS` | `10000` |

If `REQUIRE_API_KEY=0`, the API stays open (testing only). If you set `REQUIRE_API_KEY=1`, you **must** set `NMT_API_KEY` or the app will fail at startup.

---

## 4) Run and verify (copy-paste)

### Health

```bash
curl https://marconolimits-nmt.hf.space/healthz
```

### Translate (JSON — recommended)

Replace `YOUR_NMT_API_KEY` only if you enabled `REQUIRE_API_KEY=1` (otherwise omit the `-H` line).

```bash
curl -X POST "https://marconolimits-nmt.hf.space/translate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_NMT_API_KEY" \
  -d '{"text":"Hello, how are you?"}'
```

### Translate (GET — short sentences only)

```bash
curl "https://marconolimits-nmt.hf.space/translate?text=Hello"
```

### PowerShell (`Invoke-RestMethod`)

```powershell
$body = @{ text = "Hello, how are you?" } | ConvertTo-Json
$headers = @{ "X-API-Key" = "YOUR_NMT_API_KEY" }   # omit this line if API key is off
Invoke-RestMethod -Uri "https://marconolimits-nmt.hf.space/translate" -Method Post -Body $body -ContentType "application/json; charset=utf-8" -Headers $headers
```

### Production client (Python — always correct JSON)

```python
from scripts.nmt_client import translate

r = translate(
    "https://marconolimits-nmt.hf.space",
    "Hello, how are you?",
    api_key="YOUR_NMT_API_KEY",  # omit if REQUIRE_API_KEY=0
)
print(r.translation)
```

### Request formats (`POST /translate`)

- **JSON:** `Content-Type: application/json`, body `{"text":"..."}` (preferred)
- **Form:** field `text` (`application/x-www-form-urlencoded` or `multipart/form-data`)
- **Plain:** `Content-Type: text/plain`, body = the sentence only

Malformed input returns **400** with an `hint` field when applicable.

---

## 5) Free-tier expectations

- The Space can sleep when idle; the first request after sleep can be slow.
- CPU/RAM limits may cap throughput.

For heavier load or always-on behavior, consider paid Space hardware or another host.
