from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from scripts.nmt_tcp_server import load_spm, load_translator, translate_one

log = logging.getLogger("nmt_http_api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass
class AppState:
    translator: object
    sentencepiece: object
    api_key: str | None
    max_chars: int
    timeout_seconds: float
    require_api_key: bool


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class TranslateResponse(BaseModel):
    translation: str
    latency_ms: float
    request_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.getenv("MODEL_DIR", "nllb_int8")
    spm_path = os.getenv("SPM_PATH", os.path.join(model_dir, "sentencepiece.bpe.model"))
    api_key = os.getenv("NMT_API_KEY", "").strip() or None
    require_api_key = _read_env_bool("REQUIRE_API_KEY", default=api_key is not None)
    max_chars = _read_env_int("MAX_INPUT_CHARS", 2000)
    timeout_ms = _read_env_int("TRANSLATION_TIMEOUT_MS", 10000)
    timeout_seconds = max(timeout_ms / 1000.0, 0.1)

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    log.info("Loading model_dir=%s spm_path=%s", model_dir, spm_path)
    translator = load_translator(os.path.abspath(model_dir))
    sentencepiece = load_spm(os.path.abspath(spm_path))

    _ = translate_one(translator, sentencepiece, "Hi")
    log.info("Warmup complete")

    if require_api_key and not api_key:
        raise RuntimeError("REQUIRE_API_KEY is enabled but NMT_API_KEY is not set")
    if not require_api_key:
        log.warning("API key protection is disabled")

    app.state.nmt = AppState(
        translator=translator,
        sentencepiece=sentencepiece,
        api_key=api_key,
        max_chars=max_chars,
        timeout_seconds=timeout_seconds,
        require_api_key=require_api_key,
    )
    yield


app = FastAPI(title="NMT MenKan HTTP API", lifespan=lifespan)


def _require_api_key(header_key: str | None, expected_key: str | None) -> None:
    if not expected_key or not header_key or header_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/translate", response_model=TranslateResponse)
async def translate(
    request: Request,
    payload: TranslateRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> TranslateResponse:
    state: AppState = request.app.state.nmt
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))

    if state.require_api_key:
        _require_api_key(x_api_key, state.api_key)

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")
    if len(text) > state.max_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Input exceeds MAX_INPUT_CHARS={state.max_chars}",
        )

    start = time.perf_counter()
    try:
        translation = await asyncio.wait_for(
            asyncio.to_thread(translate_one, state.translator, state.sentencepiece, text),
            timeout=state.timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        log.warning("request_id=%s status=timeout", request_id)
        raise HTTPException(status_code=504, detail="Translation timed out") from exc
    except Exception:
        log.exception("request_id=%s status=error", request_id)
        raise HTTPException(status_code=500, detail="Internal server error")

    latency_ms = (time.perf_counter() - start) * 1000.0
    log.info(
        "request_id=%s status=ok chars=%d latency_ms=%.1f",
        request_id,
        len(text),
        latency_ms,
    )
    return TranslateResponse(
        translation=translation,
        latency_ms=round(latency_ms, 1),
        request_id=request_id,
    )
