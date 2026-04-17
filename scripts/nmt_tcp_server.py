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
import logging
import os
import socket
import sys
import time
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
INTER_THREADS = 8


def load_translator(model_dir: str) -> ctranslate2.Translator:
    log.info("Loading CTranslate2 from %r ...", model_dir)
    return ctranslate2.Translator(
        model_dir,
        device="cpu",
        inter_threads=INTER_THREADS,
        intra_threads=0,
    )


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


def send_http_400(conn: socket.socket) -> None:
    body = (
        "This port is the Python NMT server (raw UTF-8 lines), not HTTP.\r\n"
    ).encode("utf-8")
    hdr = (
        "HTTP/1.1 400 Bad Request\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    conn.sendall(hdr + body)


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
                log.warning("Ignoring HTTP request")
                send_http_400(conn)
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
        default="nllb_int8",
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

    # Optional: reduce OpenBLAS threading fights on Windows when using numpy elsewhere
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
