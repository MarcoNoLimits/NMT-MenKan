"""
Targeted confusion-matrix evaluation for NMT outputs.

This script builds:
1) POS confusion matrix (Expected POS vs Actual POS) from paired target words.
2) Formality confusion matrix (Expected vs Actual) from paired sentences.

Default inference uses the shipped Hugging Face CTranslate2 repo **marconolimits/en-it-nmt-ct2**
(same stack as `scripts/evaluate_nmt_fast.py`). Use `--skip-nmt` only if your CSV already
contains model predictions in the actual_* columns.

It also stores an augmented CSV with intermediate labels for analysis.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    snapshot_download = None  # type: ignore[misc, assignment]
    _HF_IMPORT_ERROR = exc
else:
    _HF_IMPORT_ERROR = None

try:
    from scripts.evaluate_nmt_fast import load_model, load_spm, translate_all
except ModuleNotFoundError:
    from evaluate_nmt_fast import load_model, load_spm, translate_all

try:
    from scripts.nmt_tcp_server import DEFAULT_SRC_LANG, DEFAULT_TGT_LANG, validate_lang_pair
except ModuleNotFoundError:
    from nmt_tcp_server import DEFAULT_SRC_LANG, DEFAULT_TGT_LANG, validate_lang_pair


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_HF_REPO = "marconolimits/en-it-nmt-ct2"


def load_spacy_model(candidates: Iterable[str]) -> spacy.language.Language:
    """Load the first available spaCy model from a candidate list."""
    for model_name in candidates:
        try:
            logging.info("Loading spaCy model: %s", model_name)
            return spacy.load(model_name)
        except OSError:
            continue
    raise OSError(
        "No usable spaCy model found. Install one of: "
        + ", ".join(candidates)
        + " (e.g. python -m spacy download en_core_web_sm)."
    )


def first_content_pos(text: str, nlp: spacy.language.Language) -> str:
    """Return POS of the first alphabetic token, or fallback to first token POS."""
    if not isinstance(text, str) or not text.strip():
        return "UNKNOWN"

    doc = nlp(text.strip())
    if not doc:
        return "UNKNOWN"

    for token in doc:
        if token.is_alpha:
            return token.pos_ or "UNKNOWN"
    return doc[0].pos_ or "UNKNOWN"


def mock_formality_classifier(text: str) -> str:
    """
    Lightweight mock formality classifier.

    Uses dictionary cues (English and Italian) to assign 'Formal' or 'Informal'.
    If tied, defaults to Formal.
    """
    if not isinstance(text, str) or not text.strip():
        return "Formal"

    lowered = text.lower()
    formal_cues = {
        "therefore",
        "however",
        "regarding",
        "kindly",
        "would you",
        "could you",
        "please",
        "sincerely",
        "dear",
        "thank you",
        "gentile",
        "cordiali saluti",
        "distinti saluti",
        "la ringrazio",
        "signore",
        "signora",
        "ebbene",
        "gradirei",
    }
    informal_cues = {
        "hey",
        "hi",
        "yo",
        "gonna",
        "wanna",
        "kinda",
        "lol",
        "bro",
        "dude",
        "pls",
        "thx",
        "ciao",
        "che figata",
        "ahahaha",
        "dai ",
        " dai",
        "tipo ",
        " amico",
        "fra ",
    }

    formal_score = sum(cue in lowered for cue in formal_cues)
    informal_score = sum(cue in lowered for cue in informal_cues)

    if informal_score > formal_score:
        return "Informal"
    return "Formal"


def first_translation_token(text: str) -> str:
    """First whitespace-separated surface token from model output (for single-word probes)."""
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.split()[0].strip(".,!?\"'()[]{};:«»")


def resolve_ct2_model_dir(hf_repo: str | None, model_dir: str | None, hf_cache_dir: str | None) -> Path:
    """Resolve local directory containing CTranslate2 artifacts."""
    if model_dir:
        path = Path(model_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"--model-dir is not a directory: {path}")
        return path.resolve()
    repo = hf_repo or DEFAULT_HF_REPO
    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required to download the model. Install it "
            "(e.g. pip install huggingface_hub) or pass --model-dir."
        ) from _HF_IMPORT_ERROR
    logging.info("Downloading / syncing Hub repo '%s' ...", repo)
    root = snapshot_download(repo_id=repo, cache_dir=hf_cache_dir, local_files_only=False)
    return Path(root)


def resolve_english_source_column(df: pd.DataFrame, preferred: str | None) -> str:
    """Pick column with English sentences to translate (EN -> IT)."""
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["source_sentence_en", "source_sentence", "english_sentence", "expected_sentence"])
    seen: set[str] = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            ordered.append(c)
            seen.add(c)
    for col in ordered:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find an English sentence column. Set --english-source-col or add one of: "
        + ", ".join(["source_sentence_en", "source_sentence", "english_sentence", "expected_sentence"])
    )


def apply_nmt_predictions(
    df: pd.DataFrame,
    *,
    model_root: Path,
    source_word_col: str,
    actual_target_word_col: str,
    actual_sentence_col: str,
    english_source_col: str,
    src_lang: str,
    tgt_lang: str,
    spm_path: str | None,
    batch_size: int,
    beam_size: int,
    inter_threads: int,
    device: str | None,
    device_index: int | None,
    translate_words: bool,
    translate_sentences: bool,
) -> pd.DataFrame:
    """Fill actual_* columns using marconolimits/en-it-nmt-ct2 (CTranslate2 + SPM)."""
    validate_lang_pair(src_lang, tgt_lang)
    mdir = str(model_root)
    spm_file = spm_path or os.path.join(mdir, "sentencepiece.bpe.model")
    if not os.path.isfile(spm_file):
        raise FileNotFoundError(f"SentencePiece model not found: {spm_file}")

    translator = load_model(mdir, inter_threads, device=device, device_index=device_index)
    sp = load_spm(spm_file)
    out = df.copy()

    if translate_sentences:
        texts = out[english_source_col].astype(str).tolist()
        logging.info("Translating %d sentences (%s -> %s)...", len(texts), src_lang, tgt_lang)
        preds = translate_all(translator, sp, texts, src_lang, tgt_lang, batch_size, beam_size)
        out[actual_sentence_col] = preds

    if translate_words:
        words = out[source_word_col].astype(str).tolist()
        logging.info("Translating %d source-word probes (%s -> %s)...", len(words), src_lang, tgt_lang)
        preds_w = translate_all(translator, sp, words, src_lang, tgt_lang, batch_size, beam_size)
        out[actual_target_word_col] = [first_translation_token(p) for p in preds_w]

    return out


def save_confusion_plot(
    confusion_df: pd.DataFrame,
    title: str,
    output_path: Path,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Render and save a high-resolution confusion matrix heatmap."""
    plt.figure(figsize=figsize)
    sns.heatmap(
        confusion_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Expected")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()
    logging.info("Saved figure: %s", output_path)


def build_pos_confusion(
    df: pd.DataFrame,
    source_word_col: str,
    expected_target_word_col: str,
    actual_target_word_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build POS confusion matrix from a dataframe.

    Returns:
        (confusion_matrix_df, augmented_dataframe)
    """
    required_cols = [source_word_col, expected_target_word_col, actual_target_word_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required POS columns: {missing}")

    source_nlp = load_spacy_model(["en_core_web_sm", "xx_sent_ud_sm"])
    target_nlp = load_spacy_model(["it_core_news_sm", "en_core_web_sm", "xx_sent_ud_sm"])

    df = df.copy()
    df["source_pos"] = df[source_word_col].apply(lambda t: first_content_pos(t, source_nlp))
    df["expected_pos"] = df[expected_target_word_col].apply(lambda t: first_content_pos(t, target_nlp))
    df["actual_pos"] = df[actual_target_word_col].apply(lambda t: first_content_pos(t, target_nlp))

    confusion = pd.crosstab(
        df["expected_pos"],
        df["actual_pos"],
        rownames=["Expected POS"],
        colnames=["Actual POS"],
        dropna=False,
    )
    return confusion, df


def build_formality_confusion(
    df: pd.DataFrame,
    expected_sentence_col: str,
    actual_sentence_col: str,
    expected_formality_col: str | None = None,
    actual_formality_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build formality confusion matrix from CSV.

    If label columns are not provided, labels are inferred using mock_formality_classifier.
    """
    needed_text_cols = [expected_sentence_col, actual_sentence_col]
    missing = [col for col in needed_text_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required formality text columns: {missing}")

    df = df.copy()
    if expected_formality_col and expected_formality_col in df.columns:
        df["expected_formality"] = df[expected_formality_col].fillna("Formal").astype(str)
    else:
        df["expected_formality"] = df[expected_sentence_col].apply(mock_formality_classifier)

    if actual_formality_col and actual_formality_col in df.columns:
        df["actual_formality"] = df[actual_formality_col].fillna("Formal").astype(str)
    else:
        df["actual_formality"] = df[actual_sentence_col].apply(mock_formality_classifier)

    allowed = {"Formal", "Informal"}
    df["expected_formality"] = df["expected_formality"].apply(
        lambda x: "Informal" if str(x).strip().lower() == "informal" else "Formal"
    )
    df["actual_formality"] = df["actual_formality"].apply(
        lambda x: "Informal" if str(x).strip().lower() == "informal" else "Formal"
    )

    confusion = pd.crosstab(
        df["expected_formality"],
        df["actual_formality"],
        rownames=["Expected Formality"],
        colnames=["Actual Formality"],
        dropna=False,
    ).reindex(index=sorted(allowed), columns=sorted(allowed), fill_value=0)
    return confusion, df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate POS and formality confusion matrices for NMT outputs."
    )
    parser.add_argument("--csv-path", required=True, help="Input CSV path.")
    parser.add_argument("--output-dir", default="reports/confusion_matrices", help="Output directory.")

    parser.add_argument(
        "--skip-nmt",
        action="store_true",
        help="Use actual_* columns from CSV only (no CTranslate2 / Hugging Face model).",
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face repo id for CTranslate2 weights (default: {DEFAULT_HF_REPO}).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Local CTranslate2 model directory (skips Hub download when set).",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for snapshot_download.",
    )
    parser.add_argument("--spm-model", default=None, help="SentencePiece path (default: <model-dir>/sentencepiece.bpe.model).")
    parser.add_argument(
        "--english-source-col",
        default=None,
        help="Column with English sentences for EN→IT (auto-detect if omitted).",
    )
    parser.add_argument("--source-lang", default=DEFAULT_SRC_LANG)
    parser.add_argument("--target-lang", default=DEFAULT_TGT_LANG)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--inter-threads", type=int, default=8)
    parser.add_argument("--device", default=None, help="CTranslate2 device (default: env NMT_DEVICE or cpu).")
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument(
        "--skip-sentence-nmt",
        action="store_true",
        help="Do not overwrite sentence-level actual_* translations (POS/word probes still run if enabled).",
    )
    parser.add_argument(
        "--skip-word-nmt",
        action="store_true",
        help="Do not overwrite actual_target_word with EN→IT translation of source_word.",
    )

    parser.add_argument("--source-word-col", default="source_word")
    parser.add_argument("--expected-target-word-col", default="expected_target_word")
    parser.add_argument("--actual-target-word-col", default="actual_target_word")

    parser.add_argument("--expected-sentence-col", default="expected_sentence")
    parser.add_argument("--actual-sentence-col", default="actual_sentence")
    parser.add_argument("--expected-formality-col", default=None)
    parser.add_argument("--actual-formality-col", default=None)
    return parser.parse_args()


def main() -> None:
    # Avoid OpenMP duplicate-runtime abort on some Windows setups (common with MKL + other libs).
    if os.name == "nt" and "KMP_DUPLICATE_LIB_OK" not in os.environ:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    args = parse_args()
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    model_note = "CSV-provided actual_* columns"

    if not args.skip_nmt:
        root = resolve_ct2_model_dir(args.hf_repo, args.model_dir, args.hf_cache_dir)
        en_col = resolve_english_source_column(df, args.english_source_col)
        translate_words = not args.skip_word_nmt
        translate_sentences = not args.skip_sentence_nmt
        if not translate_words and not translate_sentences:
            logging.warning("Both word and sentence NMT are skipped; CSV actual_* columns are unchanged.")
        model_note = args.hf_repo if not args.model_dir else f"{args.hf_repo} (local {root})"
        df = apply_nmt_predictions(
            df,
            model_root=root,
            source_word_col=args.source_word_col,
            actual_target_word_col=args.actual_target_word_col,
            actual_sentence_col=args.actual_sentence_col,
            english_source_col=en_col,
            src_lang=args.source_lang,
            tgt_lang=args.target_lang,
            spm_path=args.spm_model,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            inter_threads=args.inter_threads,
            device=args.device,
            device_index=args.device_index,
            translate_words=translate_words,
            translate_sentences=translate_sentences,
        )

    pos_conf, pos_df = build_pos_confusion(
        df,
        source_word_col=args.source_word_col,
        expected_target_word_col=args.expected_target_word_col,
        actual_target_word_col=args.actual_target_word_col,
    )
    save_confusion_plot(
        pos_conf,
        title=f"POS confusion — Expected vs Actual ({model_note})",
        output_path=output_dir / "pos_confusion_matrix.png",
    )

    formality_conf, formality_df = build_formality_confusion(
        df,
        expected_sentence_col=args.expected_sentence_col,
        actual_sentence_col=args.actual_sentence_col,
        expected_formality_col=args.expected_formality_col,
        actual_formality_col=args.actual_formality_col,
    )
    save_confusion_plot(
        formality_conf,
        title=f"Formality confusion — Expected vs Actual ({model_note})",
        output_path=output_dir / "formality_confusion_matrix.png",
        figsize=(7, 6),
    )

    merged = pos_df.join(
        formality_df[["expected_formality", "actual_formality"]],
        how="left",
    )
    merged.to_csv(output_dir / "evaluation_with_labels.csv", index=False)

    logging.info("POS confusion matrix:\n%s", pos_conf)
    logging.info("Formality confusion matrix:\n%s", formality_conf)
    logging.info("Done. Outputs are in %s", output_dir.resolve())


if __name__ == "__main__":
    main()
