"""
Automate a WSD/lexical-choice experiment for NMT models on Hugging Face.

Reads test cases from JSON, generates top-N translations per sentence, extracts
the lexical choice for ambiguous words, and writes a probability distribution CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import ctranslate2
    import sentencepiece as spm
    from huggingface_hub import snapshot_download
except ImportError:
    ctranslate2 = None  # type: ignore[misc, assignment]
    spm = None  # type: ignore[misc, assignment]
    snapshot_download = None  # type: ignore[misc, assignment]

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    torch = None  # type: ignore[misc, assignment]
    AutoModelForSeq2SeqLM = None  # type: ignore[misc, assignment]
    AutoTokenizer = None  # type: ignore[misc, assignment]

try:
    from scripts.nmt_tcp_server import (
        DEFAULT_SRC_LANG,
        DEFAULT_TGT_LANG,
        validate_lang_pair,
    )
except ModuleNotFoundError:
    from nmt_tcp_server import (
        DEFAULT_SRC_LANG,
        DEFAULT_TGT_LANG,
        validate_lang_pair,
    )


WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WSD/lexical-choice experiment and export lexical_distribution.csv",
        epilog=(
            "Default model marconolimits/en-it-nmt-ct2 uses CTranslate2 (same as production). "
            "On some Windows setups you may need: set KMP_DUPLICATE_LIB_OK=TRUE "
            "if OpenMP DLL conflicts appear."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test-cases", default="test_cases.json", help="Path to JSON test cases")
    parser.add_argument(
        "--model-id",
        default="marconolimits/en-it-nmt-ct2",
        help="Hugging Face repo id (CTranslate2 bundle or Transformers checkpoint) or local path",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ct2", "transformers"],
        default="auto",
        help="auto: CT2 if repo id/path looks like en-it-nmt-ct2 or --ct2-model-dir is set",
    )
    parser.add_argument(
        "--ct2-model-dir",
        default=None,
        help="Local CTranslate2 model directory (skips Hub download if set and valid)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="CTranslate2 device when backend is ct2 (sets NMT_DEVICE). Default: env NMT_DEVICE or cpu.",
    )
    parser.add_argument(
        "--source-lang",
        default=DEFAULT_SRC_LANG,
        help="NLLB source language tag (default eng_Latn)",
    )
    parser.add_argument(
        "--target-lang",
        default=DEFAULT_TGT_LANG,
        help="NLLB target language tag (default ita_Latn)",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["beam", "sample"],
        default="beam",
        help="Use beam search or stochastic sampling",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of translation candidates")
    parser.add_argument("--beam-size", type=int, default=10, help="Beam width in beam mode")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus threshold")
    parser.add_argument(
        "--output-csv",
        default="lexical_distribution.csv",
        help="Output CSV for lexical distribution",
    )
    return parser.parse_args()


def load_test_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("test_cases.json must contain a JSON array.")

    normalized: list[dict[str, Any]] = []
    for i, item in enumerate(payload, start=1):
        if isinstance(item, str):
            normalized.append(
                {
                    "case_id": f"case_{i}",
                    "sentence": item,
                    "ambiguous_word": "hello",
                }
            )
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Entry #{i} must be either a string or object.")
        if "sentence" not in item or "ambiguous_word" not in item:
            raise ValueError(
                f"Entry #{i} must contain at least 'sentence' and 'ambiguous_word'."
            )
        normalized.append(
            {
                "case_id": str(item.get("case_id", f"case_{i}")),
                "sentence": str(item["sentence"]),
                "ambiguous_word": str(item["ambiguous_word"]),
                "choice_candidates": item.get("choice_candidates"),
                "target_choice_pattern": item.get("target_choice_pattern"),
            }
        )
    return normalized


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def heuristic_choice_from_translation(sentence: str, ambiguous_word: str, translation: str) -> str:
    src_tokens = tokenize_words(sentence)
    tgt_tokens = tokenize_words(translation)
    if not tgt_tokens:
        return "<EMPTY>"

    if not src_tokens:
        return tgt_tokens[0]

    src_lower = [w.lower() for w in src_tokens]
    try:
        amb_idx = src_lower.index(ambiguous_word.lower())
    except ValueError:
        amb_idx = 0

    if len(src_tokens) == 1 or len(tgt_tokens) == 1:
        return tgt_tokens[0]

    projected = round((amb_idx / (len(src_tokens) - 1)) * (len(tgt_tokens) - 1))
    left = max(0, projected - 2)
    right = min(len(tgt_tokens), projected + 3)
    window = tgt_tokens[left:right]
    return window[0] if window else tgt_tokens[0]


def extract_choice(case: dict[str, Any], translation: str) -> str:
    candidates = case.get("choice_candidates")
    if isinstance(candidates, list) and candidates:
        lowered = translation.lower()
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            if re.search(rf"\b{re.escape(candidate.lower())}\b", lowered):
                return candidate

    pattern = case.get("target_choice_pattern")
    if isinstance(pattern, str) and pattern.strip():
        match = re.search(pattern, translation, flags=re.IGNORECASE)
        if match:
            return match.group(1) if match.groups() else match.group(0)

    return heuristic_choice_from_translation(
        sentence=case["sentence"],
        ambiguous_word=case["ambiguous_word"],
        translation=translation,
    )


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend
    if args.ct2_model_dir:
        return "ct2"
    mid = args.model_id.strip().lower()
    if "en-it-nmt-ct2" in mid or mid.endswith("-ct2"):
        return "ct2"
    return "transformers"


def ensure_ct2_paths(repo_id: str, ct2_model_dir: str | None) -> tuple[str, str]:
    """Return (ct2_model_directory, sentencepiece.bpe.model path)."""
    if ct2_model_dir:
        root = Path(ct2_model_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"--ct2-model-dir is not a directory: {root}")
    else:
        if snapshot_download is None:
            raise RuntimeError("Install huggingface_hub to download CT2 models from the Hub")
        root = Path(snapshot_download(repo_id=repo_id))

    spm_path = root / "sentencepiece.bpe.model"
    if not spm_path.is_file():
        found = list(root.glob("**/sentencepiece.bpe.model"))
        if not found:
            raise FileNotFoundError(f"No sentencepiece.bpe.model under {root}")
        spm_path = found[0]

    model_dir = str(spm_path.parent)
    return model_dir, str(spm_path)


def load_ct2_translator(model_dir: str) -> ctranslate2.Translator:
    device = os.environ.get("NMT_DEVICE", "cpu").strip().lower() or "cpu"
    try:
        idx = int(os.environ.get("NMT_DEVICE_INDEX", "0"))
    except ValueError:
        idx = 0
    inter_threads = int(os.environ.get("NMT_INTER_THREADS", "2"))
    intra_threads = int(os.environ.get("NMT_INTRA_THREADS", "0"))
    kwargs: dict[str, Any] = dict(
        device=device,
        inter_threads=inter_threads,
        intra_threads=intra_threads,
    )
    if device != "cpu":
        kwargs["device_index"] = idx
    return ctranslate2.Translator(model_dir, **kwargs)


def load_spm_processor(spm_path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    return sp


def tokenize_for_ct2(sp: spm.SentencePieceProcessor, text: str, src_lang: str) -> list[str]:
    tokens = sp.EncodeAsPieces(text)
    tokens.append("</s>")
    tokens.append(src_lang)
    return tokens


def decode_ct2_hypothesis(sp: spm.SentencePieceProcessor, tokens: list[str], tgt_lang: str) -> str:
    text = sp.Decode(tokens)
    if text.startswith(tgt_lang):
        text = text[len(tgt_lang) :].lstrip()
    return text


def load_model_and_tokenizer(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def maybe_set_lang_tokens(
    tokenizer: Any,
    model: Any,
    source_lang: str,
    target_lang: str,
) -> int | None:
    forced_bos_token_id = None
    if hasattr(tokenizer, "src_lang"):
        try:
            tokenizer.src_lang = source_lang
        except Exception:
            pass
    if hasattr(tokenizer, "lang_code_to_id"):
        lang_map = getattr(tokenizer, "lang_code_to_id")
        if isinstance(lang_map, dict) and target_lang in lang_map:
            forced_bos_token_id = lang_map[target_lang]
    if forced_bos_token_id is None:
        config_bos = getattr(model.config, "forced_bos_token_id", None)
        if isinstance(config_bos, int):
            forced_bos_token_id = config_bos
    return forced_bos_token_id


def generate_candidates_transformers(
    sentence: str,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    forced_bos_token_id: int | None,
) -> list[str]:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": args.top_n,
    }
    if forced_bos_token_id is not None:
        generation_kwargs["forced_bos_token_id"] = forced_bos_token_id

    if args.generation_mode == "beam":
        generation_kwargs.update(
            {
                "do_sample": False,
                "num_beams": max(args.top_n, args.beam_size),
                "early_stopping": True,
                "num_beam_groups": 1,
            }
        )
    else:
        generation_kwargs.update(
            {
                "do_sample": True,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "num_beams": 1,
            }
        )

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def generate_candidates_ct2(
    sentence: str,
    translator: ctranslate2.Translator,
    sp: spm.SentencePieceProcessor,
    args: argparse.Namespace,
    src_lang: str,
    tgt_lang: str,
) -> list[str]:
    batch_in = [tokenize_for_ct2(sp, sentence, src_lang)]
    tgt_prefix = [[tgt_lang]]
    shared: dict[str, Any] = dict(
        max_decoding_length=args.max_new_tokens,
    )
    if args.generation_mode == "beam":
        beam = max(args.top_n, args.beam_size)
        results = translator.translate_batch(
            batch_in,
            target_prefix=tgt_prefix,
            beam_size=beam,
            num_hypotheses=args.top_n,
            **shared,
        )
    else:
        results = translator.translate_batch(
            batch_in,
            target_prefix=tgt_prefix,
            beam_size=1,
            num_hypotheses=args.top_n,
            sampling_topk=args.top_k,
            sampling_topp=args.top_p,
            sampling_temperature=args.temperature,
            **shared,
        )
    hyps = results[0].hypotheses[: args.top_n]
    return [decode_ct2_hypothesis(sp, h, tgt_lang) for h in hyps]


def build_distribution_dataframe(
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    backend: str,
    tokenizer: Any | None,
    model: Any | None,
    forced_bos_token_id: int | None,
    translator: ctranslate2.Translator | None,
    sp: spm.SentencePieceProcessor | None,
    src_lang: str,
    tgt_lang: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in cases:
        if backend == "ct2":
            assert translator is not None and sp is not None
            candidates = generate_candidates_ct2(
                sentence=case["sentence"],
                translator=translator,
                sp=sp,
                args=args,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
        else:
            assert tokenizer is not None and model is not None
            candidates = generate_candidates_transformers(
                sentence=case["sentence"],
                tokenizer=tokenizer,
                model=model,
                args=args,
                forced_bos_token_id=forced_bos_token_id,
            )
        choices = [extract_choice(case, cand) for cand in candidates]
        counts = Counter(choices)
        total = sum(counts.values()) or 1

        for lexical_choice, count in counts.most_common():
            rows.append(
                {
                    "case_id": case["case_id"],
                    "source_sentence": case["sentence"],
                    "ambiguous_word": case["ambiguous_word"],
                    "lexical_choice": lexical_choice,
                    "count": int(count),
                    "probability": float(count / total),
                    "percent": round((count / total) * 100.0, 2),
                    "generation_mode": args.generation_mode,
                    "top_n": args.top_n,
                    "model_id": args.model_id,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    validate_lang_pair(args.source_lang, args.target_lang)
    cases = load_test_cases(Path(args.test_cases))
    backend = resolve_backend(args)

    tokenizer = model = None
    forced_bos_token_id = None
    translator = None
    sp = None

    if backend == "ct2":
        if ctranslate2 is None or spm is None:
            raise RuntimeError("CT2 backend requires: pip install ctranslate2 sentencepiece huggingface_hub")
        if args.device:
            os.environ["NMT_DEVICE"] = args.device
        model_dir, spm_path = ensure_ct2_paths(args.model_id, args.ct2_model_dir)
        translator = load_ct2_translator(model_dir)
        sp = load_spm_processor(spm_path)
    else:
        if torch is None or AutoTokenizer is None:
            raise RuntimeError("Transformers backend requires torch and transformers")
        tokenizer, model = load_model_and_tokenizer(args.model_id)
        forced_bos_token_id = maybe_set_lang_tokens(
            tokenizer=tokenizer,
            model=model,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )

    df = build_distribution_dataframe(
        cases=cases,
        args=args,
        backend=backend,
        tokenizer=tokenizer,
        model=model,
        forced_bos_token_id=forced_bos_token_id,
        translator=translator,
        sp=sp,
        src_lang=args.source_lang,
        tgt_lang=args.target_lang,
    )
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Saved lexical distributions to: {args.output_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
