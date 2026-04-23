"""
evaluate_dataset.py  -  Fast quantitative evaluation against custom JSONL datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import ctranslate2
import sacrebleu
import sentencepiece as spm
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_model(model_dir: str, inter_threads: int) -> ctranslate2.Translator:
    logging.info("Loading CTranslate2 model from '%s' ...", model_dir)
    return ctranslate2.Translator(
        model_dir,
        device="cpu",
        inter_threads=inter_threads,
        intra_threads=0,
    )


def tokenize_batch(sp: spm.SentencePieceProcessor, texts: list[str], src_lang: str) -> list[list[str]]:
    batch = []
    for text in texts:
        tokens = sp.EncodeAsPieces(text)
        tokens.append("</s>")
        tokens.append(src_lang)
        batch.append(tokens)
    return batch


def detokenize_and_clean(sp: spm.SentencePieceProcessor, token_sequences: list[list[str]], tgt_lang: str) -> list[str]:
    results = []
    for tokens in token_sequences:
        text = sp.Decode(tokens)
        if text.startswith(tgt_lang):
            text = text[len(tgt_lang):].lstrip()
        results.append(text)
    return results


def translate_batch(
    translator: ctranslate2.Translator,
    sp: spm.SentencePieceProcessor,
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 32,
) -> list[str]:
    predictions = []
    target_prefix = [[tgt_lang]]

    for start in tqdm(range(0, len(sentences), batch_size), desc="Translating"):
        chunk = sentences[start : start + batch_size]
        tokenized = tokenize_batch(sp, chunk, src_lang)
        tgt_prefixes = [target_prefix[0]] * len(chunk)

        results = translator.translate_batch(
            tokenized,
            target_prefix=[tgt_prefixes[i] for i in range(len(chunk))],
            beam_size=1,
            max_decoding_length=256,
        )

        raw_tokens = [r.hypotheses[0] for r in results]
        decoded = detokenize_and_clean(sp, raw_tokens, tgt_lang)
        predictions.extend(decoded)

    return predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--jsonl", required=True, help="Path to jsonl file (e.g. data/en_it_v3/test.jsonl)")
    p.add_argument("--source-lang", default="eng_Latn")
    p.add_argument("--target-lang", default="ita_Latn")
    p.add_argument("--filter-dataset", default="open_subtitles", help="Only evaluate rows from this specific dataset (default: open_subtitles)")
    p.add_argument("--max-sentences", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    spm_model = os.path.join(args.model_dir, "sentencepiece.bpe.model")

    translator = load_model(args.model_dir, 8)
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model)

    sources = []
    targets = []

    # Parse JSONL and filter
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            
            # Check if direction and dataset match
            if row.get("source_lang") == args.source_lang and row.get("target_lang") == args.target_lang:
                if row.get("dataset") == args.filter_dataset:
                    sources.append(row["source_text"])
                    targets.append(row["target_text"])
            
            if len(sources) >= args.max_sentences:
                break

    if not sources:
        logging.error("No sentences found matching %s -> %s for dataset '%s'", args.source_lang, args.target_lang, args.filter_dataset)
        return

    logging.info("Evaluating %d casual sentences from '%s'...", len(sources), args.filter_dataset)

    predictions = translate_batch(
        translator, sp, sources, args.source_lang, args.target_lang
    )

    bleu = sacrebleu.corpus_bleu(predictions, [targets])
    chrf = sacrebleu.corpus_chrf(predictions, [targets])

    print("\n" + "="*40)
    print(f"EVALUATION: {args.source_lang} -> {args.target_lang}")
    print(f"Data Filter: {args.filter_dataset}")
    print(f"Model Path: {args.model_dir}")
    print("="*40)
    print(f"BLEU Score  : {bleu.score:.2f}")
    print(f"chrF++ Score: {chrf.score:.2f}")
    print("="*40)


if __name__ == "__main__":
    main()
