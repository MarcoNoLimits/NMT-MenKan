---
title: NMT MenKan
emoji: 🌐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: En-It translation API (FastAPI + CTranslate2)
---

# NMT-MenKan

*Bridging the gap for the hearing impaired through language.*

NMT-MenKan is a bidirectional English ↔ Italian translation API hosted on Hugging Face Spaces. It runs a fine-tuned CTranslate2 model ([`marconolimits/en-it-nmt-ct2`](https://huggingface.co/marconolimits/en-it-nmt-ct2)), derived from `facebook/nllb-200-distilled-600M` and quantised to INT8, served via FastAPI.

The name **"MenKan"** comes from Bambara for *hear and understand*.

---

## Contents

- [Architecture](#architecture)
- [Supported Language Pairs](#supported-language-pairs)
- [HTTP API — Sending Requests](#http-api--sending-requests)
  - [Italian → English](#italian--english)
  - [English → Italian](#english--italian)
  - [Request formats](#request-formats)
  - [Response format](#response-format)
  - [Error codes](#error-codes)
- [Authentication](#authentication)
- [Running Locally](#running-locally)
- [Docker](#docker)
- [Hugging Face Spaces](#hugging-face-spaces)
- [LoRA Fine-Tuning](#lora-fine-tuning)
- [Evaluation Results](#evaluation-results)
- [Repository Structure](#repository-structure)

---

## Architecture

| Layer | Technology |
|-------|-----------|
| Translation model | [`marconolimits/en-it-nmt-ct2`](https://huggingface.co/marconolimits/en-it-nmt-ct2) (fine-tuned from `facebook/nllb-200-distilled-600M`) |
| Inference engine | [CTranslate2](https://github.com/OpenNMT/CTranslate2) (INT8) |
| Tokeniser | SentencePiece BPE |
| HTTP API | FastAPI + Uvicorn |
| Hosting | Hugging Face Spaces (Docker) |

The Python HTTP server handles NLLB language-tag injection (`eng_Latn` / `ita_Latn`), batched beam-search inference via CTranslate2, and response serialisation.

---

## Supported Language Pairs

| Direction | `source_lang` | `target_lang` |
|-----------|--------------|--------------|
| Italian → English | `ita_Latn` | `eng_Latn` |
| English → Italian | `eng_Latn` | `ita_Latn` |

Any other combination returns **HTTP 400**.

---

## HTTP API — Sending Requests

The API has two endpoints for translation:

- `POST /translate` — recommended for all use cases
- `GET /translate?text=...` — convenient for short strings; prefer POST for long sentences

### Italian → English

Send your Italian text with `source_lang: "ita_Latn"` and `target_lang: "eng_Latn"`.

**cURL**

```bash
curl -X POST https://marconolimits-nmt.hf.space/translate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
        "text": "Ciao, come stai?",
        "source_lang": "ita_Latn",
        "target_lang": "eng_Latn"
      }'
```

**GET shorthand**

```bash
curl "https://marconolimits-nmt.hf.space/translate?text=Ciao%2C+come+stai%3F&source_lang=ita_Latn&target_lang=eng_Latn" \
  -H "X-API-Key: YOUR_API_KEY"
```

**Python**

```python
import requests

resp = requests.post(
    "https://marconolimits-nmt.hf.space/translate",
    headers={"X-API-Key": "YOUR_API_KEY"},
    json={
        "text": "Abbiamo topi di quattro mesi che prima erano diabetici e ora non lo sono più.",
        "source_lang": "ita_Latn",
        "target_lang": "eng_Latn",
    },
)
print(resp.json()["translation"])
# → "We have four-month-old mice that used to be diabetic and now they're not."
```

---

### English → Italian

Send your English text with `source_lang: "eng_Latn"` and `target_lang: "ita_Latn"`.

**cURL**

```bash
curl -X POST https://marconolimits-nmt.hf.space/translate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
        "text": "Hello, how are you?",
        "source_lang": "eng_Latn",
        "target_lang": "ita_Latn"
      }'
```

**GET shorthand**

```bash
curl "https://marconolimits-nmt.hf.space/translate?text=Hello%2C+how+are+you%3F&source_lang=eng_Latn&target_lang=ita_Latn" \
  -H "X-API-Key: YOUR_API_KEY"
```

**Python**

```python
import requests

resp = requests.post(
    "https://marconolimits-nmt.hf.space/translate",
    headers={"X-API-Key": "YOUR_API_KEY"},
    json={
        "text": "The patient has no fever and is stable.",
        "source_lang": "eng_Latn",
        "target_lang": "ita_Latn",
    },
)
print(resp.json()["translation"])
# → "Il paziente non ha febbre ed è stabile."
```

**JavaScript / TypeScript**

```typescript
const res = await fetch("https://marconolimits-nmt.hf.space/translate", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.NMT_API_KEY ?? "",
  },
  body: JSON.stringify({
    text: "Hello, how are you?",
    source_lang: "eng_Latn",
    target_lang: "ita_Latn",
  }),
});
const data = await res.json();
console.log(data.translation);
```

---

### Request formats

`POST /translate` accepts three content types:

| Content-Type | Body example |
|-------------|-------------|
| `application/json` *(recommended)* | `{"text":"...","source_lang":"eng_Latn","target_lang":"ita_Latn"}` |
| `application/x-www-form-urlencoded` | `text=Hello&source_lang=eng_Latn&target_lang=ita_Latn` |
| `text/plain` | raw UTF-8 text (defaults to EN→IT if no lang params) |

All fields:

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `text` | string | Yes | — | 1–2000 characters |
| `source_lang` | string | No | `eng_Latn` | NLLB language tag |
| `target_lang` | string | No | `ita_Latn` | NLLB language tag |

---

### Response format

**HTTP 200**

```json
{
  "translation": "Ciao, come stai?",
  "latency_ms": 312.5,
  "request_id": "a1b2c3d4-...",
  "source_lang": "eng_Latn",
  "target_lang": "ita_Latn",
  "model_variant": "base"
}
```

---

### Error codes

| Code | Meaning |
|------|---------|
| `400` | Empty or invalid body; unsupported language pair |
| `401` | Missing or wrong `X-API-Key` |
| `413` | Input exceeds `MAX_INPUT_CHARS` (default 2000) |
| `415` | Unsupported `Content-Type` on POST |
| `422` | Validation error (e.g. `text` field absent) |
| `504` | Translation timed out (`TRANSLATION_TIMEOUT_MS`) |

---

## Authentication

When the Space is configured with `REQUIRE_API_KEY=1`, include the header on every **translation** request:

```http
X-API-Key: <your NMT_API_KEY>
```

`GET /healthz` and `GET /` never require authentication.

Store your key in an environment variable or secret store — never commit it to Git or embed it in public client binaries.

---

## Running Locally

```powershell
# Install dependencies
pip install -r requirements.txt

# Set paths (edit run_paths.env to match your local model location)
$env:MODEL_DIR  = "artifacts/ct2/en_it_v4_casual_weighted/model"
$env:SPM_PATH   = "$env:MODEL_DIR/sentencepiece.bpe.model"
$env:REQUIRE_API_KEY = "0"

# Start the API
uvicorn scripts.nmt_http_api:app --host 0.0.0.0 --port 7860 --workers 1

# Smoke test (separate terminal)
python scripts/smoke_test_http.py --base-url http://127.0.0.1:7860
```

Run FLORES-200 quality checks (model must be loaded):

```powershell
# Italian → English baseline
python scripts/evaluate_nmt_fast.py --source-lang ita_Latn --target-lang eng_Latn --reports-dir reports/baseline

# English → Italian baseline
python scripts/evaluate_nmt_fast.py --source-lang eng_Latn --target-lang ita_Latn --reports-dir reports/baseline
```

---

## Docker

```bash
# Build
docker build -t nmt-menkan .

# Run (no API key)
docker run -p 7860:7860 -e REQUIRE_API_KEY=0 nmt-menkan

# Run (with API key)
docker run -p 7860:7860 \
  -e REQUIRE_API_KEY=1 \
  -e NMT_API_KEY=my-secret-key \
  nmt-menkan
```

The Dockerfile downloads the model from `marconolimits/en-it-nmt-ct2` on Hugging Face Hub at build time.

---

## Hugging Face Spaces

The live Space is at [huggingface.co/spaces/marconolimits/NMT](https://huggingface.co/spaces/marconolimits/NMT).

Base URL: `https://marconolimits-nmt.hf.space`

For full deployment and Git push instructions see [HUGGINGFACE_SPACES.md](HUGGINGFACE_SPACES.md).  
For integration examples in multiple languages see [HF_API_INTEGRATION.md](HF_API_INTEGRATION.md).

**Free-tier notes:**
- The Space can sleep when idle; the first request after sleep may take tens of seconds.
- CPU/RAM limits apply — suitable for moderate traffic.

---

## LoRA Fine-Tuning

A LoRA adapter workflow is available for improving translation quality without touching the base model weights. The full pipeline (baseline → data prep → training → export → evaluation → deployment with rollback) is documented in [EN_IT_LORA_WORKFLOW.md](EN_IT_LORA_WORKFLOW.md).

Quick overview:

```powershell
# 1. Baseline
python scripts/evaluate_nmt_fast.py --source-lang eng_Latn --target-lang ita_Latn --reports-dir reports/baseline
python scripts/evaluate_nmt_fast.py --source-lang ita_Latn --target-lang eng_Latn --reports-dir reports/baseline

# 2. Data
python scripts/convert_model.py prepare-data --out-dir data/en_it_v1

# 3. Train
python scripts/convert_model.py train-lora --data-dir data/en_it_v1 --output-dir artifacts/lora/en_it_v1

# 4. Export to CTranslate2
python scripts/convert_model.py export-lora --adapter-dir artifacts/lora/en_it_v1/adapter --output-dir artifacts/ct2/en_it_lora_int8

# 5. Deploy with rollback switch
$env:MODEL_VARIANT = "en_it_lora"
$env:MODEL_VARIANTS_JSON = '{"base":{"model_dir":"nllb_int8","spm_path":"nllb_int8/sentencepiece.bpe.model"},"en_it_lora":{"model_dir":"artifacts/ct2/en_it_lora_int8/model","spm_path":"artifacts/ct2/en_it_lora_int8/model/sentencepiece.bpe.model"}}'
uvicorn scripts.nmt_http_api:app --host 0.0.0.0 --port 7860
```

---

## Evaluation Results

Evaluated on the **FLORES-200 devtest** corpus using direct CTranslate2 inference (batched, beam size 1).

| Direction | Date | Sentences | BLEU | chrF++ |
|-----------|------|-----------|------|--------|
| Italian → English | Apr 17 2026 | 200 | **33.68** | **61.15** |
| English → Italian | Mar 25 2026 | 1 000 | **26.92** | **56.16** |

These scores are consistent with Meta's published results for the distilled 600M model on high-resource European pairs. The translations read naturally; differences from references are mostly stylistic rather than factual.

---

## Repository Structure

```
NMT-MenKan/
├── scripts/
│   ├── nmt_http_api.py          # FastAPI HTTP server (main entry point)
│   ├── nmt_tcp_server.py        # Low-level CTranslate2 + SentencePiece wrapper
│   ├── evaluate_nmt_fast.py     # FLORES-200 BLEU/chrF evaluation
│   ├── smoke_test_http.py       # Bidirectional HTTP regression test
│   ├── convert_model.py         # Data prep, LoRA training, CTranslate2 export
│   └── hf_space_ui.html         # Browser UI served at GET /
├── Report/                      # Progress reports
├── artifacts/ct2/               # CTranslate2 model weights (Git LFS)
├── Dockerfile                   # Docker image for Hugging Face Spaces
├── requirements.txt             # Python dependencies
├── EN_IT_LORA_WORKFLOW.md       # Full LoRA fine-tuning guide
├── HUGGINGFACE_SPACES.md        # HF Spaces deployment guide
└── HF_API_INTEGRATION.md        # Multi-language integration examples
```
