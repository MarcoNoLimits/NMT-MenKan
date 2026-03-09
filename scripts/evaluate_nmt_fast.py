"""
evaluate_nmt_fast.py  –  Fast direct-Python evaluation (no TCP overhead)

Dataset : flores200 devtest (1012 professionally translated sentence pairs,
          designed specifically as the NLLB benchmark — no style mismatch)

Speed improvements over evaluate_nmt.py:
  1. Calls CTranslate2 + SentencePiece directly from Python  →  zero TCP overhead
  2. Translates in batches (BATCH_SIZE sentences at once)    →  GPU/CPU parallelism
  3. Uses inter_threads for multi-core CPU parallelism
"""

import logging
import os
import urllib.request
import tarfile
import sacrebleu
import ctranslate2
import sentencepiece as spm
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR       = "nllb_int8"                              # CTranslate2 int8 model folder
SPM_MODEL       = "nllb_int8/sentencepiece.bpe.model"      # SentencePiece tokenizer
SRC_LANG        = "eng_Latn"
TGT_LANG        = "ita_Latn"
MAX_SENTENCES   = 1000
BATCH_SIZE      = 32    # sentences per batch  (tune: 16–64 depending on RAM)
INTER_THREADS   = 8     # CPU threads for CTranslate2 (half of 16-core — leaves room for intra-op threads)
BEAM_SIZE       = 1     # 1 = greedy (fast); 2 = slightly better quality
# ──────────────────────────────────────────────────────────────────────────────


def load_model(model_dir: str, inter_threads: int) -> ctranslate2.Translator:
    logging.info(f"Loading CTranslate2 model from '{model_dir}' ...")
    translator = ctranslate2.Translator(
        model_dir,
        device="cpu",
        inter_threads=inter_threads,   # parallelism across batches
        intra_threads=0,               # let CTranslate2 auto-detect per-op threads
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


def main():
    # ── 0. Verify files exist ──────────────────────────────────────────────────
    if not os.path.isdir(MODEL_DIR):
        logging.error(f"Model directory not found: {MODEL_DIR}")
        return
    if not os.path.isfile(SPM_MODEL):
        logging.error(f"SentencePiece model not found: {SPM_MODEL}")
        return

    # ── 1. Load model & tokenizer ──────────────────────────────────────────────
    translator = load_model(MODEL_DIR, INTER_THREADS)
    sp         = load_spm(SPM_MODEL)

    # ── 2. Load flores200 devtest from the official Meta AI release ──────────
    FLORES_URL     = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
    FLORES_CACHE   = os.path.join(os.environ.get("TEMP", "/tmp"), "flores200_dataset")
    FLORES_ARCHIVE = os.path.join(os.environ.get("TEMP", "/tmp"), "flores200.tar.gz")

    if not os.path.isdir(FLORES_CACHE):
        logging.info(f"Downloading flores200 from {FLORES_URL} ...")
        urllib.request.urlretrieve(FLORES_URL, FLORES_ARCHIVE)
        logging.info("Extracting...")
        with tarfile.open(FLORES_ARCHIVE, "r:gz") as tar:
            tar.extractall(os.path.dirname(FLORES_CACHE))
        logging.info("Done.")
    else:
        logging.info(f"Using cached flores200 at {FLORES_CACHE}")

    def read_flores_lang(lang: str) -> list[str]:
        path = os.path.join(FLORES_CACHE, "devtest", f"{lang}.devtest")
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    english_sentences = read_flores_lang("eng_Latn")[:MAX_SENTENCES]
    reference_italian = read_flores_lang("ita_Latn")[:MAX_SENTENCES]

    total = len(english_sentences)
    logging.info(f"Loaded {total} flores200 devtest sentence pairs.")

    # ── 3. Translate ───────────────────────────────────────────────────────────
    predictions = translate_all(
        translator, sp,
        english_sentences,
        SRC_LANG, TGT_LANG,
        BATCH_SIZE, BEAM_SIZE,
    )

    # ── 4. Score ───────────────────────────────────────────────────────────────
    logging.info("Calculating BLEU and chrF++ scores...")
    refs  = [reference_italian]
    bleu  = sacrebleu.corpus_bleu(predictions, refs)
    chrf  = sacrebleu.corpus_chrf(predictions, refs)

    logging.info(f"BLEU Score  : {bleu.score:.2f}")
    logging.info(f"chrF++ Score: {chrf.score:.2f}")

    # ── 5. Write report ────────────────────────────────────────────────────────
    report_path = "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== NMT-MenKan English-to-Italian Accuracy Evaluation ===\n")
        f.write("Script  : evaluate_nmt_fast.py (direct CTranslate2, batched)\n")
        f.write("Dataset : flores200 devtest (eng_Latn → ita_Latn)\n")
        f.write(f"Sentences: {total}\n")
        f.write(f"Batch size: {BATCH_SIZE}   Beam size: {BEAM_SIZE}   Threads: {INTER_THREADS}\n\n")
        f.write("--- Metrics ---\n")
        f.write(f"BLEU Score  : {bleu.score:.2f}\n")
        f.write(f"{bleu.format()}\n\n")
        f.write(f"chrF++ Score: {chrf.score:.2f}\n")
        f.write(f"{chrf.format()}\n\n")
        f.write("--- Sample Outputs (First 5) ---\n")
        for i in range(min(5, total)):
            f.write(f"ENG (Source) : {english_sentences[i]}\n")
            f.write(f"ITA (Predict): {predictions[i]}\n")
            f.write(f"ITA (Target) : {reference_italian[i]}\n")
            f.write("-" * 40 + "\n")

    logging.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
