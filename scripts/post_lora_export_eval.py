"""
After train-lora completes: export merged LoRA to CTranslate2 and run fast Flores eval both directions.
Requires final_metrics.json under the LoRA output directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Export LoRA to CT2 and evaluate vs Flores (en-it both ways).")
    p.add_argument("--lora-output-dir", required=True, help="train-lora --output-dir (must contain adapter/ and final_metrics.json)")
    p.add_argument("--ct2-output-dir", required=True, help="New directory for export-lora CTranslate2 model tree")
    p.add_argument("--base-model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--quantization", default="int8")
    p.add_argument(
        "--reports-dir",
        default=None,
        help="evaluate_nmt_fast --reports-dir (default: reports/post_tune_<run_id>)",
    )
    p.add_argument(
        "--wait-for-metrics",
        action="store_true",
        help="Poll until final_metrics.json exists, then export and evaluate.",
    )
    p.add_argument("--poll-seconds", type=int, default=90)
    p.add_argument("--timeout-hours", type=int, default=12)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    lora_out = Path(args.lora_output_dir)
    if not lora_out.is_absolute():
        lora_out = (repo_root / lora_out).resolve()
    else:
        lora_out = lora_out.resolve()
    metrics = lora_out / "final_metrics.json"
    adapter = lora_out / "adapter"

    if args.wait_for_metrics:
        deadline = time.time() + args.timeout_hours * 3600
        print(f"Waiting for {metrics} ...")
        while time.time() < deadline:
            if metrics.is_file():
                break
            time.sleep(args.poll_seconds)
        else:
            print(f"Timeout waiting for {metrics}", file=sys.stderr)
            sys.exit(1)

    if not metrics.is_file():
        print(f"Missing {metrics}; train-lora must finish first.", file=sys.stderr)
        sys.exit(1)
    if not adapter.is_dir():
        print(f"Missing {adapter}.", file=sys.stderr)
        sys.exit(1)

    run_id = lora_out.name
    reports = Path(args.reports_dir) if args.reports_dir else repo_root / "reports" / f"post_tune_{run_id}"
    reports.mkdir(parents=True, exist_ok=True)

    ct2 = Path(args.ct2_output_dir)
    if not ct2.is_absolute():
        ct2 = (repo_root / ct2).resolve()
    else:
        ct2 = ct2.resolve()
    ct2.parent.mkdir(parents=True, exist_ok=True)
    convert_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "convert_model.py"),
        "export-lora",
        "--base-model",
        args.base_model,
        "--adapter-dir",
        str(adapter),
        "--output-dir",
        str(ct2),
        "--quantization",
        args.quantization,
    ]
    print("Running:", " ".join(convert_cmd))
    subprocess.check_call(convert_cmd, cwd=str(repo_root))

    model_dir = ct2 / "model"
    eval_py = repo_root / "scripts" / "evaluate_nmt_fast.py"
    for src, tgt in (("eng_Latn", "ita_Latn"), ("ita_Latn", "eng_Latn")):
        ev = [
            sys.executable,
            str(eval_py),
            "--model-dir",
            str(model_dir),
            "--spm-model",
            str(model_dir / "sentencepiece.bpe.model"),
            "--source-lang",
            src,
            "--target-lang",
            tgt,
            "--reports-dir",
            str(reports),
        ]
        print("Running:", " ".join(ev))
        subprocess.check_call(ev, cwd=str(repo_root))

    print(f"Done. CT2 model: {model_dir}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
