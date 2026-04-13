# Integrating external software with the Hugging Face NMT API

Use this when you connect **another app** (game, mobile app, desktop tool, backend service) to the **hosted** translator on Hugging Face—not your local machine.

## Production endpoint

| Item | Value |
|------|--------|
| **Base URL** | `https://marconolimits-nmt.hf.space` |
| **Protocol** | HTTPS only |
| **Space (dashboard)** | [huggingface.co/spaces/marconolimits/NMT](https://huggingface.co/spaces/marconolimits/NMT) |

All paths below are relative to the base URL (e.g. full health URL: `https://marconolimits-nmt.hf.space/healthz`).

---

## Authentication

If the Space has **`REQUIRE_API_KEY=1`** and secret **`NMT_API_KEY`** set in [Space settings](https://huggingface.co/spaces/marconolimits/NMT/settings), every **translation** request must include:

```http
X-API-Key: <your NMT_API_KEY value>
```

- **`GET /healthz`** does **not** require a key (use for monitoring).
- **`GET /`** (HTML help) does **not** require a key.
- **`POST /translate`** and **`GET /translate`** require the key when enforcement is on.

Store the key in **environment variables** or a **secure secret store** in your app—never commit it to Git or embed it in public client binaries if you can avoid it.

---

## Endpoints

### 1) Health check

```http
GET /healthz
```

**Response 200:** `{"status":"ok"}`

Use this before showing “translator online” or in uptime checks.

---

### 2) Translate (recommended: POST + JSON)

```http
POST /translate
Content-Type: application/json
X-API-Key: <optional if enforcement off>

{"text":"Your English sentence here."}
```

**Response 200:**

```json
{
  "translation": "Italian text here.",
  "latency_ms": 123.4,
  "request_id": "..."
}
```

**Typical errors**

| Code | Meaning |
|------|--------|
| 401 | Missing/wrong `X-API-Key` |
| 400 | Bad body (e.g. missing `text`, invalid JSON)—body may include `hint` |
| 413 | Text longer than `MAX_INPUT_CHARS` |
| 504 | Translation timed out |
| 415 | Unsupported `Content-Type` for POST |

---

### 3) Translate (GET, short text only)

```http
GET /translate?text=Hello
X-API-Key: <optional if enforcement off>
```

URL-encode `text` if it contains spaces or special characters. Prefer **POST** for long sentences (URL length limits).

---

## Minimal examples (copy and adapt)

### cURL

```bash
curl -sS "https://marconolimits-nmt.hf.space/healthz"

curl -sS -X POST "https://marconolimits-nmt.hf.space/translate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY_HERE" \
  -d "{\"text\":\"Hello, how are you?\"}"
```

### Python (this repo)

Use the bundled client so JSON and headers stay correct:

```python
from scripts.nmt_client import translate

r = translate(
    "https://marconolimits-nmt.hf.space",
    "Hello, how are you?",
    api_key="YOUR_KEY_HERE",  # omit if API key is disabled on the Space
)
print(r.translation)
```

### C# / .NET (`HttpClient`)

```csharp
using var client = new HttpClient { BaseAddress = new Uri("https://marconolimits-nmt.hf.space/") };
client.DefaultRequestHeaders.Add("X-API-Key", "YOUR_KEY_HERE"); // if required

var json = """{"text":"Hello, how are you?"}""";
using var content = new StringContent(json, Encoding.UTF8, "application/json");
var resp = await client.PostAsync("translate", content);
resp.EnsureSuccessStatusCode();
var body = await resp.Content.ReadAsStringAsync();
// Parse JSON: translation, latency_ms, request_id
```

### JavaScript / TypeScript (Node or server)

```javascript
const res = await fetch("https://marconolimits-nmt.hf.space/translate", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.NMT_API_KEY, // if required
  },
  body: JSON.stringify({ text: "Hello, how are you?" }),
});
const data = await res.json();
// data.translation
```

---

## Calling from a **browser** (web app)

Browsers enforce **CORS**. This API does not add permissive CORS headers by default, so **direct `fetch()` from a random website to `marconolimits-nmt.hf.space` may be blocked**.

**Recommended:** call the API from **your backend** (same origin as your web app), and have the frontend talk to your backend. Your backend adds `X-API-Key` and never exposes the key to the browser.

If you must call from the browser only, you may need a **proxy** or **CORS configuration** on a server you control—not covered here.

---

## Operational notes

- **Cold start:** Free Spaces can **sleep**. The first request after idle may take **tens of seconds**; retries with backoff help.
- **HTTPS:** Always use `https://` (certificate is managed by Hugging Face).
- **Limits:** Respect `MAX_INPUT_CHARS` (default 2000 unless changed in Space env).
- **Same key everywhere:** Use the **same** `NMT_API_KEY` value you configured in the Space **Variables and secrets** panel.

---

## Checklist for a new integration

1. Confirm the Space is **Running** on the [Space page](https://huggingface.co/spaces/marconolimits/NMT).
2. Confirm **`NMT_API_KEY`** / **`REQUIRE_API_KEY`** in Space settings match what your app sends.
3. Implement **POST `/translate`** with **`Content-Type: application/json`** and body **`{"text":"..."}`**.
4. Send **`X-API-Key`** when enforcement is on.
5. Parse JSON response fields **`translation`**, **`latency_ms`**, **`request_id`**.
6. For production web UIs, prefer **backend-to-Hugging Face**, not **browser-to-Hugging Face**, unless you have solved CORS and key secrecy.

For Space-specific Git push and env setup, see [HUGGINGFACE_SPACES.md](HUGGINGFACE_SPACES.md).
