"""
Generate GraduationReport/figures/pipeline_translation_finetune_blue_green.png
Blue: preparation / encoding / data stages. Green: decoding / export / output.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, xy, w, h, text, facecolor, textcolor="white", fontsize=9):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#1a1a1a",
        facecolor=facecolor,
        mutation_aspect=0.6,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="600",
        color=textcolor,
        wrap=True,
    )
    return patch


def _arrow(ax, start, end, color="#444444"):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)


def main():
    # Accessible blue / green (WCAG-friendly contrast on white for legend)
    blue = "#1B6CA8"
    blue_light = "#3490D4"
    green = "#1B7F5A"
    green_light = "#2EB87C"

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.2),
        dpi=150,
        gridspec_kw={"height_ratios": [1.05, 0.95], "hspace": 0.28},
    )
    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 10)
        ax.axis("off")

    # --- Top: inference ---
    ax_top.set_ylim(0, 3.2)
    ax_top.text(
        0.05,
        2.95,
        "Inference: translation pipeline (NMT-MenKan)",
        fontsize=12,
        fontweight="700",
        color="#111",
    )
    ax_top.text(
        0.05,
        2.72,
        "Blue: preprocessing & encoder   |   Green: decoder & response",
        fontsize=9,
        color="#333",
        style="italic",
    )

    bh, bw = 0.62, 1.35
    y_row = 1.35

    # Blue row: API -> tokenize+tags -> encoder
    _box(ax_top, (0.35, y_row), bw, bh, "HTTP / validate\nFastAPI", blue)
    _box(ax_top, (2.05, y_row), bw, bh, "SentencePiece\n+ NLLB src tags", blue_light)
    _box(ax_top, (3.75, y_row), bw * 1.1, bh, "Encoder\n(context)", blue)

    # Green row: decoder -> clean -> response
    y_row2 = 0.35
    _box(ax_top, (5.55, y_row2), bw * 1.15, bh, "Decoder\n(beam, tgt tag)", green)
    _box(ax_top, (7.35, y_row2), bw * 0.95, bh, "Detokenize\n+ tag cleanup", green_light)
    _box(ax_top, (8.85, y_row2), bw * 0.95, bh, "JSON\nresponse", green)

    _arrow(ax_top, (1.7, y_row + bh / 2), (2.05, y_row + bh / 2))
    _arrow(ax_top, (3.4, y_row + bh / 2), (3.75, y_row + bh / 2))
    # Encoder -> decoder (down and across)
    _arrow(ax_top, (4.85, y_row), (6.1, y_row2 + bh))
    # Along green row
    _arrow(ax_top, (6.7, y_row2 + bh / 2), (7.35, y_row2 + bh / 2))
    _arrow(ax_top, (8.3, y_row2 + bh / 2), (8.85, y_row2 + bh / 2))

    ax_top.text(
        4.95,
        0.92,
        "cross-attn",
        fontsize=8,
        color="#555",
        rotation=52,
    )

    # --- Bottom: fine-tuning ---
    ax_bot.set_ylim(0, 2.85)
    ax_bot.text(
        0.05,
        2.55,
        "Fine-tuning: adaptation before CTranslate2 serving",
        fontsize=12,
        fontweight="700",
        color="#111",
    )
    ax_bot.text(
        0.05,
        2.32,
        "Blue: data & formatting   |   Green: training, export, validation",
        fontsize=9,
        color="#333",
        style="italic",
    )

    y = 0.85
    bw2, bh2 = 1.45, 0.68
    _box(ax_bot, (0.3, y), bw2, bh2, "Parallel pairs\n(clean / domain)", blue)
    _box(ax_bot, (2.15, y), bw2 * 1.05, bh2, "NLLB tag format\n(train = infer)", blue_light)
    _box(ax_bot, (4.15, y), bw2 * 1.0, bh2, "Supervised\nfine-tune", green)
    _box(ax_bot, (6.05, y), bw2 * 1.15, bh2, "Export\nCT2 INT8", green_light)
    _box(ax_bot, (8.05, y), bw2 * 1.2, bh2, "FLORES + probes\nre-eval", green)

    _arrow(ax_bot, (1.75, y + bh2 / 2), (2.15, y + bh2 / 2))
    _arrow(ax_bot, (3.68, y + bh2 / 2), (4.15, y + bh2 / 2))
    _arrow(ax_bot, (5.62, y + bh2 / 2), (6.05, y + bh2 / 2))
    _arrow(ax_bot, (7.68, y + bh2 / 2), (8.05, y + bh2 / 2))

    legend_elems = [
        mpatches.Patch(facecolor=blue, edgecolor="#1a1a1a", label="Blue track: prep / encode / data"),
        mpatches.Patch(facecolor=green, edgecolor="#1a1a1a", label="Green track: decode / train / ship"),
    ]
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
    )

    out = Path(__file__).resolve().parents[1] / "GraduationReport" / "figures" / "pipeline_translation_finetune_blue_green.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
