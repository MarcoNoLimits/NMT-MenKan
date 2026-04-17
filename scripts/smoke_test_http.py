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


def _token_overlap(a: str, b: str) -> int:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    return len(a_tokens & b_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test NMT HTTP API endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860", help="Base URL of API service")
    parser.add_argument("--api-key", default=None, help="Optional API key for X-API-Key header")
    parser.add_argument("--text-en", default="Hello from smoke test.", help="English text to translate")
    parser.add_argument("--text-it", default="Ciao dal test di fumo.", help="Italian text to translate")
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
        body={
            "text": args.text_en,
            "source_lang": "eng_Latn",
            "target_lang": "ita_Latn",
        },
        api_key=args.api_key,
    )
    print(f"[translate en->it] status={trans_status} body={trans_body}")
    if trans_status != 200:
        print("Translate EN->IT check failed.")
        return 2

    trans_status, trans_body = _request_json(
        f"{base}/translate",
        "POST",
        body={
            "text": args.text_it,
            "source_lang": "ita_Latn",
            "target_lang": "eng_Latn",
        },
        api_key=args.api_key,
    )
    print(f"[translate it->en] status={trans_status} body={trans_body}")
    if trans_status != 200:
        print("Translate IT->EN check failed.")
        return 3

    invalid_status, invalid_body = _request_json(
        f"{base}/translate",
        "POST",
        body={
            "text": args.text_en,
            "source_lang": "eng_Latn",
            "target_lang": "fra_Latn",
        },
        api_key=args.api_key,
    )
    print(f"[translate invalid pair] status={invalid_status} body={invalid_body}")
    if invalid_status != 400:
        print("Invalid pair check failed.")
        return 4

    # Round-trip sanity check: EN->IT->EN should preserve at least some lexical overlap.
    seed_text = "The patient has no fever and is stable."
    en_to_it_status, en_to_it_body = _request_json(
        f"{base}/translate",
        "POST",
        body={
            "text": seed_text,
            "source_lang": "eng_Latn",
            "target_lang": "ita_Latn",
        },
        api_key=args.api_key,
    )
    if en_to_it_status != 200:
        print("Round-trip step EN->IT failed.")
        return 5
    en_to_it_payload = json.loads(en_to_it_body)
    it_text = en_to_it_payload.get("translation", "")

    it_to_en_status, it_to_en_body = _request_json(
        f"{base}/translate",
        "POST",
        body={
            "text": it_text,
            "source_lang": "ita_Latn",
            "target_lang": "eng_Latn",
        },
        api_key=args.api_key,
    )
    if it_to_en_status != 200:
        print("Round-trip step IT->EN failed.")
        return 6
    it_to_en_payload = json.loads(it_to_en_body)
    roundtrip = it_to_en_payload.get("translation", "")
    overlap = _token_overlap(seed_text, roundtrip)
    print(f"[roundtrip] overlap_tokens={overlap} text={roundtrip!r}")
    if overlap < 2:
        print("Round-trip quality gate failed.")
        return 7

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
