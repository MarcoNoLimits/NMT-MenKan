# NLP and LoRA Training — NMT-MenKan

Standalone summary of the project’s natural language processing stack and the LoRA adaptation workflow (not tied to the main graduation report structure).

---

The natural-language-processing core of NMT-MenKan is **bidirectional neural machine translation** between English and Italian, implemented as a **Transformer-based sequence-to-sequence** model in the **NLLB** family: source text is segmented with **SentencePiece** subword tokenization, conditioned on **NLLB language tags** so decoding stays in the requested direction, and translated with **autoregressive beam search** (typically a narrow beam for latency) before detokenization. **LoRA (Low-Rank Adaptation)** is the main supervised adaptation path: the full pretrained **encoder–decoder weights stay frozen**, and only small **rank-16** trainable matrices are injected into the attention **query, key, value, and output** projections (`r=16`, `lora_alpha=32`, `lora_dropout=0.05`), trained with the Hugging Face **Seq2SeqTrainer** on **curated parallel JSONL** data (built from sources such as **OPUS Books, Europarl, Tatoeba, and OpenSubtitles**) with **BLEU (SacreBLEU) and chrF** computed on decoded generations when evaluation is enabled. After training, the adapter is **merged into the base weights**, exported to an **INT8-quantized CTranslate2** runtime artifact, and wired into serving as a **named model variant** next to the baseline so adaptation improves domain phrasing and pair-specific behavior without abandoning the stable multilingual backbone or instant rollback to the unfine-tuned model.

---

**Related repo docs:** `EN_IT_LORA_WORKFLOW.md`, `scripts/convert_model.py` (`train-lora`, `export-lora`).
