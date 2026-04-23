from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from scripts.nmt_tcp_server import (
    DEFAULT_SRC_LANG,
    DEFAULT_TGT_LANG,
    SUPPORTED_PAIRS,
    load_spm,
    load_translator,
    translate_one,
    validate_lang_pair,
)

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
    model_variant: str


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    source_lang: str = Field(default=DEFAULT_SRC_LANG, min_length=1, max_length=32)
    target_lang: str = Field(default=DEFAULT_TGT_LANG, min_length=1, max_length=32)


class TranslateResponse(BaseModel):
    translation: str
    latency_ms: float
    request_id: str
    source_lang: str
    target_lang: str
    model_variant: str


def _resolve_model_paths() -> tuple[str, str, str]:
    model_variant = os.getenv("MODEL_VARIANT", "base").strip() or "base"
    variants_json = os.getenv("MODEL_VARIANTS_JSON", "").strip()
    if not variants_json:
        model_dir = os.getenv("MODEL_DIR", "artifacts/ct2/en_it_v4_casual_weighted/model")
        spm_path = os.getenv("SPM_PATH", os.path.join(model_dir, "sentencepiece.bpe.model"))
        return model_variant, model_dir, spm_path

    try:
        variant_map = json.loads(variants_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MODEL_VARIANTS_JSON must be valid JSON") from exc
    if model_variant not in variant_map:
        raise RuntimeError(f"MODEL_VARIANT {model_variant!r} not found in MODEL_VARIANTS_JSON")
    selected = variant_map[model_variant]
    model_dir = selected.get("model_dir")
    spm_path = selected.get("spm_path") or os.path.join(model_dir, "sentencepiece.bpe.model")
    if not model_dir:
        raise RuntimeError(f"Variant {model_variant!r} is missing model_dir")
    return model_variant, model_dir, spm_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_variant, model_dir, spm_path = _resolve_model_paths()
    api_key = os.getenv("NMT_API_KEY", "").strip() or None
    require_api_key = _read_env_bool("REQUIRE_API_KEY", default=api_key is not None)
    max_chars = _read_env_int("MAX_INPUT_CHARS", 2000)
    timeout_ms = _read_env_int("TRANSLATION_TIMEOUT_MS", 10000)
    timeout_seconds = max(timeout_ms / 1000.0, 0.1)

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    log.info(
        "Loading model_variant=%s model_dir=%s spm_path=%s",
        model_variant,
        model_dir,
        spm_path,
    )
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
        model_variant=model_variant,
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
    source_lang = payload.source_lang.strip()
    target_lang = payload.target_lang.strip()
    try:
        validate_lang_pair(source_lang, target_lang)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{exc}. Supported pairs: "
                f"{', '.join([f'{s}->{t}' for s, t in sorted(SUPPORTED_PAIRS)])}"
            ),
        ) from exc

    start = time.perf_counter()
    try:
        translation = await asyncio.wait_for(
            asyncio.to_thread(
                translate_one,
                state.translator,
                state.sentencepiece,
                text,
                source_lang,
                target_lang,
            ),
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
        source_lang=source_lang,
        target_lang=target_lang,
        model_variant=state.model_variant,
    )
