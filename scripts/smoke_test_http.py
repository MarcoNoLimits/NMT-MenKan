from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _request_json(url: str, method: str = "GET", body: dict | None = None, api_key: str | None = None) -> tuple[int, str]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["X-API-Key"] = api_key

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            return resp.status, payload
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", errors="replace")
        return e.code, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test NMT HTTP API endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860", help="Base URL of API service")
    parser.add_argument("--api-key", default=None, help="Optional API key for X-API-Key header")
    parser.add_argument("--text", default="Hello from smoke test.", help="Text to translate")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    health_status, health_body = _request_json(f"{base}/healthz", "GET")
    print(f"[healthz] status={health_status} body={health_body}")
    if health_status != 200:
        print("Health check failed.")
        return 1

    trans_status, trans_body = _request_json(
        f"{base}/translate",
        "POST",
        body={"text": args.text},
        api_key=args.api_key,
    )
    print(f"[translate] status={trans_status} body={trans_body}")
    if trans_status != 200:
        print("Translate check failed.")
        return 2

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
