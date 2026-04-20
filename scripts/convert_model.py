from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

import ctranslate2


def convert_base_model(model_name: str, output_dir: str, quantization: str) -> None:
    print(f"Converting model {model_name} to {output_dir} with {quantization} quantization...")
    converter = ctranslate2.converters.TransformersConverter(
        model_name,
        copy_files=["tokenizer.json", "sentencepiece.bpe.model"],
    )
    converter.convert(output_dir, quantization=quantization, force=True)
    print(f"Model saved to {os.path.abspath(output_dir)}")


def prepare_data(out_dir: str, seed: int, val_ratio: float, test_ratio: float, max_per_corpus: int) -> None:
    import warnings
    from datasets import load_dataset
    from datasets import concatenate_datasets

    warnings.filterwarnings(
        "ignore",
        message=r".*Helsinki-NLP/tatoeba_mt contains custom code.*",
        category=FutureWarning,
    )

    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def extract_value(item: dict, key: str) -> str:
        value = item
        for part in key.split("."):
            value = value[part]
        return value

    def to_rows(ds, src_key: str, tgt_key: str, name: str) -> list[dict]:
        rows = []
        for item in ds:
            src = normalize(extract_value(item, src_key))
            tgt = normalize(extract_value(item, tgt_key))
            if not src or not tgt:
                continue
            rows.append({"source_text": src, "target_text": tgt, "source_lang": "eng_Latn", "target_lang": "ita_Latn", "dataset": name})
            rows.append({"source_text": tgt, "target_text": src, "source_lang": "ita_Latn", "target_lang": "eng_Latn", "dataset": name})
        return rows

    def load_tatoeba_rows():
        """
        Handle split changes in newer datasets releases where tatoeba_mt may expose
        validation/test only instead of train.
        """
        try:
            return load_dataset("Helsinki-NLP/tatoeba_mt", "eng-ita", split="train")
        except ValueError:
            validation = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-ita", split="validation")
            test = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-ita", split="test")
            return concatenate_datasets([validation, test])

    books = load_dataset("opus_books", "en-it", split="train")
    europarl = load_dataset("Helsinki-NLP/europarl", "en-it", split="train")
    tatoeba = load_tatoeba_rows()
    books = books.select(range(min(len(books), max_per_corpus)))
    europarl = europarl.select(range(min(len(europarl), max_per_corpus)))
    tatoeba = tatoeba.select(range(min(len(tatoeba), max_per_corpus)))

    rows = []
    rows.extend(to_rows(books, "translation.en", "translation.it", "opus_books"))
    rows.extend(to_rows(europarl, "translation.en", "translation.it", "europarl"))
    rows.extend(to_rows(tatoeba, "sourceString", "targetString", "tatoeba"))

    deduped = []
    seen = set()
    for row in rows:
        token_len_a = len(row["source_text"].split())
        token_len_b = len(row["target_text"].split())
        if token_len_a < 2 or token_len_b < 2 or token_len_a > 120 or token_len_b > 120:
            continue
        if max(token_len_a, token_len_b) / max(1, min(token_len_a, token_len_b)) > 3.0:
            continue
        key = f"{row['source_lang']}|{row['target_lang']}|{row['source_text']}|{row['target_text']}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        deduped.append(row)

    rng = random.Random(seed)
    rng.shuffle(deduped)
    n_total = len(deduped)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    test_rows = deduped[:n_test]
    val_rows = deduped[n_test : n_test + n_val]
    train_rows = deduped[n_test + n_val :]

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    def write_jsonl(name: str, samples: list[dict]) -> None:
        with (path / name).open("w", encoding="utf-8") as f:
            for row in samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl("train.jsonl", train_rows)
    write_jsonl("val.jsonl", val_rows)
    write_jsonl("test.jsonl", test_rows)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "datasets": ["opus_books", "europarl", "tatoeba"],
                "seed": seed,
                "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows), "total": n_total},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote curated dataset to {path}")


def train_lora(
    data_dir: str,
    output_dir: str,
    model_name: str,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    num_train_epochs: float,
    resume_from_checkpoint: str | None,
) -> None:
    import evaluate
    import numpy as np
    import warnings
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

    warnings.filterwarnings(
        "ignore",
        message=r".*Trainer\.tokenizer is now deprecated.*",
        category=FutureWarning,
    )

    dataset = load_dataset(
        "json",
        data_files={"train": str(Path(data_dir) / "train.jsonl"), "validation": str(Path(data_dir) / "val.jsonl")},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    def preprocess(batch: dict) -> dict:
        tokenized = {"input_ids": [], "attention_mask": [], "labels": []}
        for src_text, tgt_text, src_lang, tgt_lang in zip(
            batch["source_text"], batch["target_text"], batch["source_lang"], batch["target_lang"]
        ):
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang
            inputs = tokenizer(src_text, max_length=max_length, truncation=True)
            labels = tokenizer(text_target=tgt_text, max_length=max_length, truncation=True)
            tokenized["input_ids"].append(inputs["input_ids"])
            tokenized["attention_mask"].append(inputs["attention_mask"])
            tokenized["labels"].append(labels["input_ids"])
        return tokenized

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute_metrics(eval_preds: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu = bleu_metric.compute(predictions=decoded_preds, references=[[x] for x in decoded_labels])["score"]
        chrf = chrf_metric.compute(predictions=decoded_preds, references=[[x] for x in decoded_labels])["score"]
        return {"bleu": round(bleu, 2), "chrf": round(chrf, 2)}

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(Path(output_dir) / "checkpoints"),
        learning_rate=2e-4,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        predict_with_generate=True,
        report_to="none",
        metric_for_best_model="bleu",
        greater_is_better=True,
        load_best_model_at_end=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate()
    adapter_dir = Path(output_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (Path(output_dir) / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter to {adapter_dir}")


def export_lora(base_model: str, adapter_dir: str, output_dir: str, quantization: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged_hf = target / "merged_hf"
    merged.save_pretrained(merged_hf)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(merged_hf)

    converter = ctranslate2.converters.TransformersConverter(str(merged_hf))
    converter.convert(str(target / "model"), quantization=quantization)
    print(f"Exported CTranslate2 model to {target / 'model'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NMT utility script for conversion/data/LoRA workflows.")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert-base")
    c.add_argument("--model-name", default="facebook/nllb-200-distilled-600M")
    c.add_argument("--output-dir", default="nllb_int8")
    c.add_argument("--quantization", default="int8")

    d = sub.add_parser("prepare-data")
    d.add_argument("--out-dir", default="data/en_it_v1")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--val-ratio", type=float, default=0.05)
    d.add_argument("--test-ratio", type=float, default=0.05)
    d.add_argument("--max-per-corpus", type=int, default=120000)

    t = sub.add_parser("train-lora")
    t.add_argument("--data-dir", default="data/en_it_v1")
    t.add_argument("--output-dir", default="artifacts/lora/en_it_v1")
    t.add_argument("--model-name", default="facebook/nllb-200-distilled-600M")
    t.add_argument("--train-batch-size", type=int, default=8)
    t.add_argument("--eval-batch-size", type=int, default=8)
    t.add_argument("--gradient-accumulation-steps", type=int, default=1)
    t.add_argument("--max-length", type=int, default=192)
    t.add_argument("--num-train-epochs", type=float, default=2.0)
    t.add_argument("--resume-from-checkpoint", default=None)

    e = sub.add_parser("export-lora")
    e.add_argument("--base-model", default="facebook/nllb-200-distilled-600M")
    e.add_argument("--adapter-dir", required=True)
    e.add_argument("--output-dir", default="artifacts/ct2/en_it_lora_int8")
    e.add_argument("--quantization", default="int8")

    args = parser.parse_args()
    if args.command == "convert-base":
        convert_base_model(args.model_name, args.output_dir, args.quantization)
    elif args.command == "prepare-data":
        prepare_data(args.out_dir, args.seed, args.val_ratio, args.test_ratio, args.max_per_corpus)
    elif args.command == "train-lora":
        train_lora(
            args.data_dir,
            args.output_dir,
            args.model_name,
            args.train_batch_size,
            args.eval_batch_size,
            args.gradient_accumulation_steps,
            args.max_length,
            args.num_train_epochs,
            args.resume_from_checkpoint,
        )
    else:
        export_lora(args.base_model, args.adapter_dir, args.output_dir, args.quantization)


if __name__ == "__main__":
    main()
