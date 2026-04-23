"""
Lightweight LoRA training monitor: GPU stats + checkpoint / trainer_state progress.
Appends to a log file every N seconds. Logs a stale warning if step does not advance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def nvidia_smi_snapshot() -> str:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            return f"nvidia-smi failed: {r.stderr or r.stdout}"
        return (r.stdout or "").strip().replace("\n", " | ")
    except FileNotFoundError:
        return "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        return "nvidia-smi timeout"


def latest_checkpoint_info(checkpoints_root: Path) -> tuple[str | None, float | None]:
    if not checkpoints_root.is_dir():
        return None, None
    best: tuple[float, Path] | None = None
    for p in checkpoints_root.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, p)
    if best is None:
        return None, None
    return best[1].name, best[0]


def read_trainer_global_step(checkpoints_root: Path) -> int | None:
    state_paths = sorted(
        checkpoints_root.glob("checkpoint-*/trainer_state.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for sp in state_paths[:1]:
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
            step = data.get("global_step")
            if isinstance(step, int):
                return step
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor LoRA training (GPU + checkpoints).")
    parser.add_argument("--output-dir", required=True, help="train-lora --output-dir (contains checkpoints/)")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Append log path (default: reports/training_monitor_<run_id>.log)",
    )
    parser.add_argument("--interval-seconds", type=int, default=120)
    parser.add_argument(
        "--stale-intervals",
        type=int,
        default=6,
        help="If global_step unchanged for this many intervals, log a stale warning.",
    )
    parser.add_argument("--run-id", default=None, help="Short id for default log filename (folder name if omitted).")
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    run_id = args.run_id or out.name
    log_file = Path(args.log_file) if args.log_file else Path("reports") / f"training_monitor_{run_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    ck_root = out / "checkpoints"
    last_step: int | None = None
    stale_count = 0

    while True:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        gpu = nvidia_smi_snapshot()
        ck_name, ck_mtime = latest_checkpoint_info(ck_root)
        step = read_trainer_global_step(ck_root)
        ck_part = f"checkpoint={ck_name or 'none'} mtime={ck_mtime}"
        step_part = f"global_step={step if step is not None else 'n/a'}"

        line = f"{ts} | {gpu} | {ck_part} | {step_part}\n"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line)
        print(line, end="", flush=True)

        if step is not None:
            if last_step is not None and step == last_step:
                stale_count += 1
            else:
                stale_count = 0
            last_step = step
            if stale_count >= args.stale_intervals:
                warn = (
                    f"{ts} | STALE: global_step unchanged for {stale_count} intervals "
                    f"(threshold {args.stale_intervals}); training may be stuck.\n"
                )
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(warn)
                print(warn, end="", flush=True)
                stale_count = 0

        if (out / "final_metrics.json").is_file():
            done = f"{ts} | DONE: final_metrics.json present; monitor exiting.\n"
            with log_file.open("a", encoding="utf-8") as f:
                f.write(done)
            print(done, end="", flush=True)
            break

        time.sleep(max(5, args.interval_seconds))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
