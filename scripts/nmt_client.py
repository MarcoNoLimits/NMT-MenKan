"""
Production-friendly client for the NMT HTTP API.

Always sends POST /translate with Content-Type: application/json and body {"text": "..."}.
Use this from services, jobs, and tests to avoid format mistakes.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TranslateResult:
    translation: str
    latency_ms: float
    request_id: str
    raw: dict[str, Any]


def translate(
    base_url: str,
    text: str,
    *,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    request_id: str | None = None,
) -> TranslateResult:
    """
    Translate English text to Italian via POST /translate.

    base_url: e.g. "https://marconolimits-nmt.hf.space" (no trailing slash)
    api_key: sent as X-API-Key when set (match Space secrets).
    """
    root = base_url.rstrip("/")
    url = f"{root}/translate"
    payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    }
    if api_key:
        headers["X-API-Key"] = api_key
    if request_id:
        headers["X-Request-Id"] = request_id

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            raw_body = resp.read().decode("utf-8")
            data = json.loads(raw_body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_body}") from e

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response: {data!r}")
    if "translation" not in data:
        raise RuntimeError(f"Missing translation in response: {data!r}")

    return TranslateResult(
        translation=str(data["translation"]),
        latency_ms=float(data.get("latency_ms", 0.0)),
        request_id=str(data.get("request_id", "")),
        raw=data,
    )


def healthz(base_url: str, *, timeout_s: float = 30.0) -> bool:
    root = base_url.rstrip("/")
    req = urllib.request.Request(f"{root}/healthz", method="GET")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            return resp.status == 200
    except OSError:
        return False
