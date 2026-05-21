"""
NMT TCP server in Python — same CTranslate2 + SentencePiece path as evaluate_nmt_fast.py.

Use this instead of NMT_MenKan.exe when the native build falls back to float32 or hits
OpenBLAS issues; the pip ctranslate2 wheel usually has efficient int8.

Protocol (matches C++ server):
  - Listen on TCP (default port 18080)
  - One request per connection: UTF-8 line ending with \\n
  - Reply: UTF-8 Italian text (no trailing newline required)
  - HTTP request lines (GET / …) get a 400 response and are not translated

Run from the directory that contains the `nllb_int8` folder (e.g. build\\Release):

  pip install ctranslate2 sentencepiece
  python scripts/nmt_tcp_server.py

Or:

  python scripts/nmt_tcp_server.py --model-dir C:/path/to/nllb_int8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import socket
import sys
import time
import urllib.parse
from typing import TYPE_CHECKING

import ctranslate2
import sentencepiece as spm

if TYPE_CHECKING:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SRC_LANG = "eng_Latn"
DEFAULT_TGT_LANG = "ita_Latn"
SUPPORTED_PAIRS = {
    ("eng_Latn", "ita_Latn"),
    ("ita_Latn", "eng_Latn"),
}
BEAM_SIZE = 1
MAX_DECODE = 256
# inter_threads > vCPU_count causes core contention and slows everything down.
INTER_THREADS = int(os.environ.get("NMT_INTER_THREADS", "1"))
INTRA_THREADS = int(os.environ.get("NMT_INTRA_THREADS", "0"))


def _ct2_device() -> tuple[str, int]:
    """Inference device: default cpu. Set NMT_DEVICE=cuda for GPU (requires CUDA build of ctranslate2)."""
    device = os.environ.get("NMT_DEVICE", "cpu").strip().lower() or "cpu"
    try:
        idx = int(os.environ.get("NMT_DEVICE_INDEX", "0"))
    except ValueError:
        idx = 0
    return device, idx


def load_translator(model_dir: str) -> ctranslate2.Translator:
    device, device_index = _ct2_device()
    log.info(
        "Loading CTranslate2 from %r device=%s index=%s inter_threads=%d intra_threads=%d",
        model_dir, device, device_index, INTER_THREADS, INTRA_THREADS,
    )
    kwargs: dict = dict(
        device=device,
        inter_threads=INTER_THREADS,
        intra_threads=INTRA_THREADS,
    )
    if device != "cpu":
        kwargs["device_index"] = device_index
    return ctranslate2.Translator(model_dir, **kwargs)


def load_spm(path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def tokenize_line(sp: spm.SentencePieceProcessor, text: str, src_lang: str) -> list[str]:
    tokens = sp.EncodeAsPieces(text)
    tokens.append("</s>")
    tokens.append(src_lang)
    return tokens


def validate_lang_pair(src_lang: str, tgt_lang: str) -> None:
    pair = (src_lang, tgt_lang)
    if pair not in SUPPORTED_PAIRS:
        raise ValueError(
            f"Unsupported language pair {src_lang}->{tgt_lang}. "
            "Allowed pairs: eng_Latn->ita_Latn, ita_Latn->eng_Latn"
        )


def translate_one(
    translator: ctranslate2.Translator,
    sp: spm.SentencePieceProcessor,
    text: str,
    src_lang: str = DEFAULT_SRC_LANG,
    tgt_lang: str = DEFAULT_TGT_LANG,
) -> str:
    validate_lang_pair(src_lang, tgt_lang)
    batch_in = [tokenize_line(sp, text, src_lang)]
    results = translator.translate_batch(
        batch_in,
        target_prefix=[[tgt_lang]],
        beam_size=BEAM_SIZE,
        max_decoding_length=MAX_DECODE,
    )
    raw = results[0].hypotheses[0]
    out = sp.Decode(raw)
    if out.startswith(tgt_lang):
        out = out[len(tgt_lang) :].lstrip()
    return out


def looks_like_http(line: str) -> bool:
    u = line[:12].upper()
    return (
        u.startswith("GET ")
        or u.startswith("POST ")
        or u.startswith("PUT ")
        or u.startswith("HEAD ")
        or u.startswith("DELETE ")
        or u.startswith("OPTIONS ")
    )


def recv_line(conn: socket.socket, max_len: int = 256 * 1024) -> bytes:
    buf = bytearray()
    while len(buf) < max_len:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf.extend(chunk)
        if b"\n" in buf:
            break
        if len(chunk) < 4096:
            break
    return bytes(buf)


def recv_http_request(conn: socket.socket, initial_bytes: bytes) -> tuple[str, dict[str, str], bytes]:
    # Read until headers end (\r\n\r\n or \n\n)
    data = initial_bytes
    while b"\r\n\r\n" not in data and b"\n\n" not in data:
        try:
            chunk = conn.recv(4096)
        except Exception:
            break
        if not chunk:
            break
        data += chunk
    
    # Split headers and body
    if b"\r\n\r\n" in data:
        headers_part, body_part = data.split(b"\r\n\r\n", 1)
    elif b"\n\n" in data:
        headers_part, body_part = data.split(b"\n\n", 1)
    else:
        headers_part = data
        body_part = b""
        
    lines = headers_part.decode("utf-8", errors="replace").splitlines()
    if not lines:
        return "", {}, b""
        
    req_line = lines[0]
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
            
    # Read the rest of the body based on Content-Length
    try:
        content_length = int(headers.get("content-length", "0"))
    except ValueError:
        content_length = 0

    while len(body_part) < content_length:
        try:
            chunk = conn.recv(4096)
        except Exception:
            break
        if not chunk:
            break
        body_part += chunk
        
    if len(body_part) > content_length:
        body_part = body_part[:content_length]
        
    return req_line, headers, body_part


def send_http_response(conn: socket.socket, status_code: int, status_text: str, content_type: str, body: bytes) -> None:
    hdr = (
        f"HTTP/1.1 {status_code} {status_text}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    try:
        conn.sendall(hdr + body)
    except Exception as e:
        log.warning("Failed to send HTTP response: %s", e)


def serve(
    host: str,
    port: int,
    model_dir: str,
    spm_path: str,
    default_src_lang: str,
    default_tgt_lang: str,
) -> None:
    if not os.path.isdir(model_dir):
        log.error("Model directory not found: %s", model_dir)
        sys.exit(1)
    if not os.path.isfile(spm_path):
        log.error("SentencePiece model not found: %s", spm_path)
        sys.exit(1)

    translator = load_translator(model_dir)
    sp = load_spm(spm_path)
    # Warmup (same idea as C++ server)
    _ = translate_one(
        translator,
        sp,
        "Hi",
        src_lang=default_src_lang,
        tgt_lang=default_tgt_lang,
    )
    log.info("Warmup done.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    log.info("Listening on %s:%s (Python / pip ctranslate2)", host, port)

    while True:
        conn, addr = sock.accept()
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            raw = recv_line(conn)
            if not raw:
                continue
            line = raw.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
            if not line:
                continue
            log.info("Received from %s: %s", addr, line[:200])

            if looks_like_http(line):
                log.info("Handling HTTP request tolerantly on TCP port")
                req_line, headers, body_part = recv_http_request(conn, raw)
                
                parts = req_line.split()
                if len(parts) < 2:
                    send_http_response(conn, 400, "Bad Request", "text/plain; charset=utf-8", b"Invalid HTTP request line")
                    continue
                    
                method = parts[0].upper()
                url_path = parts[1]
                
                parsed_url = urllib.parse.urlparse(url_path)
                params = urllib.parse.parse_qs(parsed_url.query)
                
                text = ""
                src_lang = default_src_lang
                tgt_lang = default_tgt_lang
                
                if method == "GET":
                    text = params.get("text", [""])[0]
                    src_lang = params.get("source_lang", [default_src_lang])[0]
                    tgt_lang = params.get("target_lang", [default_tgt_lang])[0]
                elif method == "POST":
                    ct = headers.get("content-type", "").lower()
                    if "application/json" in ct:
                        try:
                            body_json = json.loads(body_part.decode("utf-8", errors="replace"))
                            text = body_json.get("text", "")
                            src_lang = body_json.get("source_lang", default_src_lang)
                            tgt_lang = body_json.get("target_lang", default_tgt_lang)
                        except Exception as e:
                            log.warning("Failed to parse HTTP JSON body: %s", e)
                    elif "application/x-www-form-urlencoded" in ct:
                        try:
                            body_str = body_part.decode("utf-8", errors="replace")
                            body_params = urllib.parse.parse_qs(body_str)
                            text = body_params.get("text", [""])[0]
                            src_lang = body_params.get("source_lang", [default_src_lang])[0]
                            tgt_lang = body_params.get("target_lang", [default_tgt_lang])[0]
                        except Exception as e:
                            log.warning("Failed to parse HTTP form body: %s", e)
                    else:
                        text = body_part.decode("utf-8", errors="replace").strip()
                        if not text:
                            text = params.get("text", [""])[0]
                            src_lang = params.get("source_lang", [default_src_lang])[0]
                            tgt_lang = params.get("target_lang", [default_tgt_lang])[0]
                
                if not text:
                    if parsed_url.path in ("/", "/healthz"):
                        send_http_response(conn, 200, "OK", "application/json; charset=utf-8", b'{"status":"ok"}')
                    else:
                        send_http_response(conn, 400, "Bad Request", "text/plain; charset=utf-8", b"Missing 'text' parameter")
                    continue
                
                try:
                    validate_lang_pair(src_lang, tgt_lang)
                except ValueError as exc:
                    send_http_response(conn, 400, "Bad Request", "text/plain; charset=utf-8", str(exc).encode("utf-8"))
                    continue
                    
                t0_http = time.perf_counter()
                translated = translate_one(
                    translator,
                    sp,
                    text,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                )
                ms_http = (time.perf_counter() - t0_http) * 1000.0
                log.info("HTTP request translated in %.1f ms", ms_http)
                
                accept = headers.get("accept", "").lower()
                if "application/json" in accept or "json" in url_path:
                    resp_data = {
                        "translation": translated,
                        "latency_ms": round(ms_http, 1),
                        "source_lang": src_lang,
                        "target_lang": tgt_lang
                    }
                    body_bytes = json.dumps(resp_data, ensure_ascii=False).encode("utf-8")
                    send_http_response(conn, 200, "OK", "application/json; charset=utf-8", body_bytes)
                else:
                    send_http_response(conn, 200, "OK", "text/plain; charset=utf-8", translated.encode("utf-8"))
                continue

            t0 = time.perf_counter()
            translated = translate_one(
                translator,
                sp,
                line,
                src_lang=default_src_lang,
                tgt_lang=default_tgt_lang,
            )
            ms = (time.perf_counter() - t0) * 1000.0
            log.info("Translated in %.1f ms", ms)
            out = translated.encode("utf-8")
            conn.sendall(out)
        except Exception as e:
            log.exception("Request failed: %s", e)
        finally:
            conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Python TCP NMT server (evaluate_nmt_fast stack)")
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=18080, help="TCP port")
    p.add_argument(
        "--model-dir",
        default="artifacts/ct2/en_it_v4_casual_weighted/model",
        help="Path to CTranslate2 model directory (relative to cwd or absolute)",
    )
    p.add_argument(
        "--spm",
        default=None,
        help="Path to sentencepiece.bpe.model (default: <model-dir>/sentencepiece.bpe.model)",
    )
    p.add_argument(
        "--src-lang",
        default=DEFAULT_SRC_LANG,
        help="Default source language for TCP text-only protocol",
    )
    p.add_argument(
        "--tgt-lang",
        default=DEFAULT_TGT_LANG,
        help="Default target language for TCP text-only protocol",
    )
    args = p.parse_args()
    validate_lang_pair(args.src_lang, args.tgt_lang)
    spm_path = args.spm or os.path.join(args.model_dir, "sentencepiece.bpe.model")

    serve(
        args.host,
        args.port,
        os.path.abspath(args.model_dir),
        os.path.abspath(spm_path),
        args.src_lang,
        args.tgt_lang,
    )


if __name__ == "__main__":
    main()
