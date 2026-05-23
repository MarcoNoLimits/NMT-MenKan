"""
Reproducible significance tests for NMT-MenKan experiments.

Typical uses:
  - Categorical diagnostics (confusion matrices): Pearson chi-square and Cramer's V,
    Fisher's exact test for 2x2 tables.
  - Continuous MT metrics: paired bootstrap on sentence-level chrF++ (or BLEU) from
    `evaluate_nmt_fast.py --sentence-metrics-out`.
  - Bidirectional comparison: Wilcoxon signed-rank on index-aligned sentence chrF++
    (same FLORES line id, opposite translation directions).

Examples:
  python scripts/compute_experiment_statistics.py confusion-tables
  python scripts/compute_experiment_statistics.py paired-bootstrap \\
      --metrics-a reports/stats/model_a_eng_ita.jsonl \\
      --metrics-b reports/stats/legacy_eng_ita.jsonl --metric chrf
  python scripts/compute_experiment_statistics.py paired-directions \\
      --jsonl-en-it reports/stats/a_en_it.jsonl --jsonl-it-en reports/stats/a_it_en.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, wilcoxon

# ---------------------------------------------------------------------------
# Contingency tables from Graduation_Report §4.2 (N = 5,000, model A, EN→IT probes)
# Rows: expected POS / expected register; columns: model-derived labels.
# ---------------------------------------------------------------------------

POS_TABLE = np.array(
    [
        [495, 0, 965, 0, 0],  # ADJ expected
        [0, 487, 0, 511, 0],  # ADV expected
        [0, 0, 1049, 0, 0],  # NOUN expected
        [0, 0, 0, 0, 1493],  # VERB expected
    ],
    dtype=np.int64,
)
POS_ROW_LABELS = ("ADJ", "ADV", "NOUN", "VERB")
POS_COL_LABELS = ("ADJ", "ADV", "NOUN", "PUNCT", "VERB")

FORMALITY_TABLE = np.array([[2215, 731], [517, 1537]], dtype=np.int64)


def _cramers_v(chi2: float, n: int, nrows: int, ncols: int) -> float:
    k = min(nrows - 1, ncols - 1)
    if k <= 0 or n <= 0:
        return float("nan")
    return math.sqrt(chi2 / (n * k))


def analyze_contingency(table: np.ndarray, title: str) -> dict:
    chi2, p, dof, expected = chi2_contingency(table)
    n = int(table.sum())
    v = _cramers_v(chi2, n, table.shape[0], table.shape[1])
    return {
        "title": title,
        "chi2": float(chi2),
        "df": int(dof),
        "p_value": float(p),
        "cramers_v": float(v),
        "n": n,
        "expected": expected.tolist(),
    }


def analyze_fisher_2x2(table: np.ndarray, title: str) -> dict:
    oddsratio, p = fisher_exact(table)
    return {"title": title, "odds_ratio": float(oddsratio), "p_value_two_sided": float(p)}


def load_metrics_jsonl(path: Path, metric: str) -> np.ndarray:
    scores: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scores.append(float(rec[metric]))
    return np.asarray(scores, dtype=np.float64)


def paired_bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap on mean(a - b); returns a percentile CI (no p-value)."""
    if a.shape != b.shape:
        raise ValueError("Arrays must be the same length for paired bootstrap.")
    d = a - b
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    obs = float(np.mean(d))
    boots = np.empty(n_resamples, dtype=np.float64)
    idx = np.arange(n)
    for i in range(n_resamples):
        sample_idx = rng.choice(idx, size=n, replace=True)
        boots[i] = float(np.mean(d[sample_idx]))
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff_a_minus_b": obs,
        "bootstrap_mean_diff_ci95": (float(ci_low), float(ci_high)),
        "n_resamples": n_resamples,
        "n_sentences": n,
    }


def paired_wilcoxon_report(x: np.ndarray, y: np.ndarray, *, alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank on paired samples (e.g., sentence chrF IT→EN vs EN→IT)."""
    stat, p = wilcoxon(x, y, alternative=alternative, zero_method="wilcox", correction=False)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_pairs": int(len(x)),
        "alternative": alternative,
    }


def cmd_confusion_tables() -> int:
    pos = analyze_contingency(POS_TABLE, "POS expected vs actual (report section 4.2)")
    form_chi = analyze_contingency(FORMALITY_TABLE, "Formality expected vs actual (report section 4.2)")
    form_fish = analyze_fisher_2x2(FORMALITY_TABLE, "Formality 2x2 Fisher exact")

    lines = [
        "=== Confusion-matrix statistics (fixed counts from report section 4.2) ===",
        "",
        f"[{pos['title']}]",
        f"  Pearson chi-square = {pos['chi2']:.4f}, df = {pos['df']}, p = {pos['p_value']:.4g}",
        f"  Cramer's V = {pos['cramers_v']:.4f} (large association for N = {pos['n']})",
        f"  Row labels: {POS_ROW_LABELS}; column labels: {POS_COL_LABELS}",
        "",
        f"[{form_chi['title']}] (independence)",
        f"  Pearson chi-square = {form_chi['chi2']:.4f}, df = {form_chi['df']}, p = {form_chi['p_value']:.4g}",
        f"  Cramer's V = {form_chi['cramers_v']:.4f}",
        "",
        f"[{form_fish['title']}]",
        f"  Odds ratio = {form_fish['odds_ratio']:.4f}, two-sided p = {form_fish['p_value_two_sided']:.4g}",
        "",
        "Interpretation: p-values test independence of row and column classifications.",
        "With N = 5,000, minute departures from independence yield p ~ 0; report Cramer's V alongside p.",
    ]
    print("\n".join(lines))
    return 0


def cmd_paired_bootstrap(ns: argparse.Namespace) -> int:
    a = load_metrics_jsonl(Path(ns.metrics_a), ns.metric)
    b = load_metrics_jsonl(Path(ns.metrics_b), ns.metric)
    w = paired_wilcoxon_report(a, b, alternative="two-sided")
    r = paired_bootstrap_mean_diff(a, b, n_resamples=ns.n_resamples, seed=ns.seed)
    print("=== Paired comparison (metrics A vs B, same sentence order) ===")
    print(json.dumps({"wilcoxon_signed_rank": w, "paired_bootstrap_mean_a_minus_b": r, "metric": ns.metric}, indent=2))
    return 0


def cmd_paired_directions(ns: argparse.Namespace) -> int:
    en_it = load_metrics_jsonl(Path(ns.jsonl_en_it), ns.metric)
    it_en = load_metrics_jsonl(Path(ns.jsonl_it_en), ns.metric)
    m = min(len(en_it), len(it_en))
    en_it = en_it[:m]
    it_en = it_en[:m]
    # Test whether IT→EN scores differ from EN→IT on the same parallel line index.
    w = paired_wilcoxon_report(it_en, en_it, alternative=ns.alternative)
    boot = paired_bootstrap_mean_diff(it_en, en_it, n_resamples=ns.n_resamples, seed=ns.seed)
    print("=== Paired direction comparison (same FLORES index) ===")
    print(json.dumps({"wilcoxon_it_en_minus_en_it": w, "bootstrap_mean_it_en_minus_en_it_ci": boot}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Statistical tests for NMT-MenKan experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("confusion-tables", help="Chi-square / Fisher on section 4.2 confusion tables.")

    pb = sub.add_parser("paired-bootstrap", help="Paired bootstrap on sentence metrics JSONL.")
    pb.add_argument("--metrics-a", required=True, type=Path)
    pb.add_argument("--metrics-b", required=True, type=Path)
    pb.add_argument("--metric", choices=("chrf", "bleu"), default="chrf")
    pb.add_argument("--n-resamples", type=int, default=10_000)
    pb.add_argument("--seed", type=int, default=0)
    pb.set_defaults(func=cmd_paired_bootstrap)

    pd = sub.add_parser(
        "paired-directions",
        help="Wilcoxon + bootstrap on IT→EN vs EN→IT sentence metrics (aligned JSONL).",
    )
    pd.add_argument("--jsonl-en-it", required=True, type=Path)
    pd.add_argument("--jsonl-it-en", required=True, type=Path)
    pd.add_argument("--metric", choices=("chrf", "bleu"), default="chrf")
    pd.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
        help="Wilcoxon alternative: 'greater' tests IT→EN > EN→IT.",
    )
    pd.add_argument("--n-resamples", type=int, default=10_000)
    pd.add_argument("--seed", type=int, default=0)
    pd.set_defaults(func=cmd_paired_directions)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = build_parser()
    ns = p.parse_args(argv)
    if ns.cmd == "confusion-tables":
        return cmd_confusion_tables()
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
