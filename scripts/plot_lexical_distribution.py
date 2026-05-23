"""
Create report-ready figure from lexical_distribution.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot lexical-choice distributions for report.")
    parser.add_argument("--input-csv", default="lexical_distribution.csv", help="Input CSV path")
    parser.add_argument(
        "--output-figure",
        default="GraduationReport/figures/wsd_lexical_confidence_en_it.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--title",
        default="WSD lexical confidence by ambiguous word (beam top-10)",
        help="Chart title",
    )
    parser.add_argument(
        "--x-label",
        default="Average top lexical-choice probability (%)",
        help="X-axis label",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    # Keep strongest lexical choice per test case (top-1 probability),
    # then aggregate by ambiguous word for a cleaner summary view.
    top1 = (
        df.sort_values(["case_id", "probability"], ascending=[True, False])
        .groupby("case_id", as_index=False)
        .first()
    )
    agg = (
        top1.groupby("ambiguous_word", as_index=False)
        .agg(avg_top1_probability=("probability", "mean"))
        .sort_values("avg_top1_probability", ascending=False)
        .reset_index(drop=True)
    )

    labels = agg["ambiguous_word"].tolist()
    scores = (agg["avg_top1_probability"] * 100.0).tolist()

    plt.figure(figsize=(10, 5.5))
    bars = plt.barh(range(len(labels)), scores, color="#2D7FF9")
    plt.gca().invert_yaxis()
    plt.yticks(range(len(labels)), labels, fontsize=10)
    plt.xlim(0, 100)
    plt.xlabel(args.x_label, fontsize=10)
    plt.title(args.title, fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.25)

    for bar, score in zip(bars, scores):
        plt.text(
            score + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%",
            va="center",
            fontsize=9,
        )

    out_path = Path(args.output_figure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
