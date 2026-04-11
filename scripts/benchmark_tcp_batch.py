"""
Send many English lines from a text file to the NMT TCP server (one request per line),
measure round-trip time for each translation, and write a report.

Usage (with server on default port 18080):
  python scripts/benchmark_tcp_batch.py
  python scripts/benchmark_tcp_batch.py --file path/to/sentences.txt
  python scripts/benchmark_tcp_batch.py --host 127.0.0.1 --port 18080 --out results.tsv
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path


def recv_until_close(sock: socket.socket, max_bytes: int = 4 * 1024 * 1024) -> bytes:
    """Server sends one reply then closes; read until EOF."""
    chunks: list[bytes] = []
    total = 0
    while total < max_bytes:
        chunk = sock.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def translate_one(text: str, host: str, port: int, timeout: float) -> tuple[str, float]:
    t0 = time.perf_counter()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((host, port))
        s.sendall(text.encode("utf-8") + b"\n")
        raw = recv_until_close(s)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    try:
        out = raw.decode("utf-8")
    except UnicodeDecodeError:
        out = raw.decode("utf-8", errors="replace")
    return out, elapsed_ms


def load_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_file = root / "scripts" / "sample_sentences_en.txt"

    p = argparse.ArgumentParser(description="Benchmark NMT TCP server sentence-by-sentence.")
    p.add_argument(
        "--file",
        "-f",
        type=Path,
        default=default_file,
        help="UTF-8 text file: one English sentence per line (# = comment)",
    )
    p.add_argument("--host", default="127.0.0.1", help="NMT server host")
    p.add_argument("--port", type=int, default=18080, help="NMT server port")
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Write TSV report (default: scripts/benchmark_tcp_results.tsv)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request socket timeout in seconds",
    )
    args = p.parse_args()

    if not args.file.is_file():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    sentences = load_lines(args.file)
    if not sentences:
        print("No sentences to translate (empty file or only comments).", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    if out_path is None:
        out_path = root / "scripts" / "benchmark_tcp_results.tsv"

    rows: list[tuple[int, str, str, float]] = []
    print(f"Loaded {len(sentences)} lines from {args.file}")
    print(f"Connecting to {args.host}:{args.port} …\n")

    for i, line in enumerate(sentences, start=1):
        try:
            italian, ms = translate_one(line, args.host, args.port, args.timeout)
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            rows.append((i, line, f"ERROR: {e}", -1.0))
            continue
        rows.append((i, line, italian, ms))
        print(f"[{i}] {ms:8.1f} ms  |  EN: {line[:80]}{'…' if len(line) > 80 else ''}")
        print(f"        IT: {italian[:120]}{'…' if len(italian) > 120 else ''}\n")

    times = [r[3] for r in rows if r[3] >= 0]
    if times:
        print(
            f"---\nDone. Per-sentence times: min {min(times):.1f} ms, "
            f"max {max(times):.1f} ms, avg {sum(times) / len(times):.1f} ms"
        )

    # TSV: index, ms, english, italian (tabs escaped in fields not needed if we quote carefully)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("index\tms\tenglish\titalian\n")
        for idx, en, it, ms in rows:
            en_esc = en.replace("\t", " ").replace("\n", " ")
            it_esc = it.replace("\t", " ").replace("\n", " ")
            f.write(f"{idx}\t{ms:.3f}\t{en_esc}\t{it_esc}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
