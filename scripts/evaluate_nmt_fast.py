"""
evaluate_nmt_fast.py  -  Fast direct-Python evaluation (no TCP overhead).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
import urllib.request
from datetime import datetime
from pathlib import Path

import ctranslate2
import sacrebleu
import sentencepiece as spm
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None  # type: ignore[misc, assignment]

try:
    from scripts.nmt_tcp_server import DEFAULT_SRC_LANG, DEFAULT_TGT_LANG, validate_lang_pair
except ModuleNotFoundError:
    from nmt_tcp_server import DEFAULT_SRC_LANG, DEFAULT_TGT_LANG, validate_lang_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def load_model(model_dir: str, inter_threads: int, device: str | None = None, device_index: int | None = None) -> ctranslate2.Translator:
    dev = (device or os.environ.get("NMT_DEVICE", "cpu")).strip().lower() or "cpu"
    if device_index is None:
        try:
            idx = int(os.environ.get("NMT_DEVICE_INDEX", "0"))
        except ValueError:
            idx = 0
    else:
        idx = device_index
    logging.info("Loading CTranslate2 model from '%s' (device=%s index=%s) ...", model_dir, dev, idx)
    if dev == "cpu":
        translator = ctranslate2.Translator(
            model_dir,
            device=dev,
            inter_threads=inter_threads,
            intra_threads=0,
        )
    else:
        translator = ctranslate2.Translator(
            model_dir,
            device=dev,
            device_index=idx,
            inter_threads=inter_threads,
            intra_threads=0,
        )
    logging.info("Model loaded.")
    return translator


def load_spm(spm_path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    return sp


def tokenize_batch(sp: spm.SentencePieceProcessor, texts: list[str], src_lang: str) -> list[list[str]]:
    """Tokenize a list of sentences and append NLLB language tags."""
    batch = []
    for text in texts:
        tokens = sp.EncodeAsPieces(text)
        tokens.append("</s>")
        tokens.append(src_lang)
        batch.append(tokens)
    return batch


def detokenize_and_clean(sp: spm.SentencePieceProcessor, token_sequences: list[list[str]], tgt_lang: str) -> list[str]:
    """Decode token sequences and strip the leading language tag."""
    results = []
    for tokens in token_sequences:
        text = sp.Decode(tokens)
        if text.startswith(tgt_lang):
            text = text[len(tgt_lang):].lstrip()
        results.append(text)
    return results


def translate_all(
    translator: ctranslate2.Translator,
    sp: spm.SentencePieceProcessor,
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    beam_size: int,
) -> list[str]:
    predictions = []
    target_prefix = [[tgt_lang]]  # reused for every sentence in a batch

    for start in tqdm(range(0, len(sentences), batch_size), desc="Translating batches"):
        chunk = sentences[start : start + batch_size]

        tokenized = tokenize_batch(sp, chunk, src_lang)
        tgt_prefixes = [target_prefix[0]] * len(chunk)

        results = translator.translate_batch(
            tokenized,
            target_prefix=[tgt_prefixes[i] for i in range(len(chunk))],
            beam_size=beam_size,
            max_decoding_length=256,
        )

        raw_tokens = [r.hypotheses[0] for r in results]
        decoded = detokenize_and_clean(sp, raw_tokens, tgt_lang)
        predictions.extend(decoded)

    return predictions


def load_flores(lang: str, max_sentences: int, flores_cache: str) -> list[str]:
    path = os.path.join(flores_cache, "devtest", f"{lang}.devtest")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()][:max_sentences]


def ensure_flores_cache() -> str:
    flores_cache = os.path.join(os.environ.get("TEMP", "/tmp"), "flores200_dataset")
    flores_archive = os.path.join(os.environ.get("TEMP", "/tmp"), "flores200.tar.gz")
    devtest_ok = os.path.isfile(os.path.join(flores_cache, "devtest", "eng_Latn.devtest"))
    if not os.path.isdir(flores_cache) or not devtest_ok:
        if os.path.isdir(flores_cache) and not devtest_ok:
            logging.warning("FLORES cache at %s is incomplete; re-downloading.", flores_cache)
        logging.info("Downloading flores200 from %s ...", FLORES_URL)
        urllib.request.urlretrieve(FLORES_URL, flores_archive)
        logging.info("Extracting...")
        with tarfile.open(flores_archive, "r:gz") as tar:
            tar.extractall(os.path.dirname(flores_cache))
        logging.info("Done.")
    else:
        logging.info("Using cached flores200 at %s", flores_cache)
    return flores_cache


def resolve_flores_root(flores_cache: str) -> str:
    """Return the directory whose child is `devtest/` containing FLORES language files."""
    if os.path.isfile(os.path.join(flores_cache, "devtest", "eng_Latn.devtest")):
        return flores_cache
    for root, dirs, _files in os.walk(flores_cache):
        if "devtest" not in dirs:
            continue
        dt = os.path.join(root, "devtest")
        if os.path.isfile(os.path.join(dt, "eng_Latn.devtest")):
            logging.info("Resolved FLORES devtest under %s", root)
            return root
    return flores_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast BLEU/chrF evaluator for En<->It.")
    p.add_argument(
        "--model-dir",
        default="artifacts/ct2/en_it_v4_casual_weighted/model",
        help="Path to converted CT2 model",
    )
    p.add_argument(
        "--device",
        default=None,
        help="CTranslate2 device (default: env NMT_DEVICE or cpu). Use cuda for GPU inference.",
    )
    p.add_argument("--device-index", type=int, default=None, help="GPU index when device is cuda (default: env NMT_DEVICE_INDEX or 0).")
    p.add_argument("--spm-model", default=None, help="Default: <model-dir>/sentencepiece.bpe.model")
    p.add_argument("--source-lang", default=DEFAULT_SRC_LANG)
    p.add_argument("--target-lang", default=DEFAULT_TGT_LANG)
    p.add_argument("--max-sentences", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--inter-threads", type=int, default=8)
    p.add_argument("--beam-size", type=int, default=1)
    p.add_argument("--reports-dir", default="reports/baseline")
    p.add_argument(
        "--sentence-metrics-out",
        default=None,
        help="Optional path to write JSONL with per-sentence chrF++ and smoothed sentence BLEU for statistics.",
    )
    p.add_argument(
        "--hf-repo",
        default=None,
        metavar="REPO_ID",
        help="If set, download/sync this Hugging Face CTranslate2 repo and use it as --model-dir (overrides --model-dir).",
    )
    p.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Cache directory for --hf-repo snapshot_download (default: env HF_HOME or ./.hf_cache).",
    )
    return p.parse_args()


def resolve_hf_repo_model_dir(repo_id: str, cache_dir: str | None) -> str:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required for --hf-repo (pip install huggingface_hub).")
    cd = cache_dir or os.environ.get("HF_HOME") or os.path.join(os.getcwd(), ".hf_cache")
    root = snapshot_download(repo_id=repo_id, cache_dir=cd)
    return root


def main() -> None:
    args = parse_args()
    validate_lang_pair(args.source_lang, args.target_lang)
    model_dir = resolve_hf_repo_model_dir(args.hf_repo, args.hf_cache_dir) if args.hf_repo else args.model_dir
    spm_model = args.spm_model or os.path.join(model_dir, "sentencepiece.bpe.model")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not os.path.isfile(spm_model):
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model}")

    translator = load_model(model_dir, args.inter_threads, device=args.device, device_index=args.device_index)
    sp = load_spm(spm_model)
    flores_cache = resolve_flores_root(ensure_flores_cache())

    source_sentences = load_flores(args.source_lang, args.max_sentences, flores_cache)
    target_references = load_flores(args.target_lang, args.max_sentences, flores_cache)
    total = min(len(source_sentences), len(target_references))
    source_sentences = source_sentences[:total]
    target_references = target_references[:total]
    logging.info("Loaded %d flores200 devtest sentence pairs.", total)

    predictions = translate_all(
        translator,
        sp,
        source_sentences,
        args.source_lang,
        args.target_lang,
        args.batch_size,
        args.beam_size,
    )

    logging.info("Calculating BLEU and chrF++ scores...")
    refs = [target_references]
    bleu = sacrebleu.corpus_bleu(predictions, refs)
    chrf = sacrebleu.corpus_chrf(predictions, refs)

    logging.info("BLEU Score  : %.2f", bleu.score)
    logging.info("chrF++ Score: %.2f", chrf.score)

    if args.sentence_metrics_out:
        chrf_metric = CHRF(word_order=2, beta=2)
        bleu_metric = BLEU(effective_order=True)
        out_path = Path(args.sentence_metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as jf:
            for i, (hyp, ref) in enumerate(zip(predictions, target_references)):
                s_chrf = float(chrf_metric.sentence_score(hyp, [ref]).score)
                s_bleu = float(bleu_metric.sentence_score(hyp, [ref]).score)
                rec = {
                    "i": i,
                    "source_lang": args.source_lang,
                    "target_lang": args.target_lang,
                    "chrf": s_chrf,
                    "bleu": s_bleu,
                }
                jf.write(json.dumps(rec) + "\n")
        logging.info("Wrote sentence-level metrics for %d lines to %s", total, out_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pair_name = f"{args.source_lang}_to_{args.target_lang}"
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    txt_path = reports_dir / f"{timestamp}_{pair_name}.txt"
    json_path = reports_dir / f"{timestamp}_{pair_name}.json"

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=== NMT-MenKan Accuracy Evaluation ===\n")
        f.write("Script: evaluate_nmt_fast.py (direct CTranslate2, batched)\n")
        f.write(f"Dataset: flores200 devtest ({args.source_lang} -> {args.target_lang})\n")
        f.write(f"Sentences: {total}\n")
        f.write(
            f"Batch size: {args.batch_size}   Beam size: {args.beam_size}   "
            f"Threads: {args.inter_threads}\n\n"
        )
        f.write("--- Metrics ---\n")
        f.write(f"BLEU Score  : {bleu.score:.2f}\n")
        f.write(f"{bleu.format()}\n\n")
        f.write(f"chrF++ Score: {chrf.score:.2f}\n")
        f.write(f"{chrf.format()}\n\n")
        f.write("--- Sample Outputs (First 5) ---\n")
        for i in range(min(5, total)):
            f.write(f"SRC: {source_sentences[i]}\n")
            f.write(f"PRED: {predictions[i]}\n")
            f.write(f"REF: {target_references[i]}\n")
            f.write("-" * 40 + "\n")

    payload = {
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "dataset": "flores200/devtest",
        "sentences": total,
        "batch_size": args.batch_size,
        "beam_size": args.beam_size,
        "inter_threads": args.inter_threads,
        "bleu": bleu.score,
        "chrf": chrf.score,
        "text_report": str(txt_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Reports saved to %s and %s", txt_path, json_path)


if __name__ == "__main__":
    main()
