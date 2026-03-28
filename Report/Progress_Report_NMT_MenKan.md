# NMT-MenKan — Progress Report
**Date:** 25 March 2026
**Project:** Real-Time Neural Machine Translation for HoloLens 2

---

## Overview

NMT-MenKan is an on-device translation engine built for the Microsoft HoloLens 2. Its job is to receive English text from an upstream speech recogniser and return Italian text in real time, displayed as holographic subtitles. The name "MenKan" comes from Bambara, meaning *hear and understand*. The engine runs entirely on-device using the `facebook/nllb-200-distilled-600M` model, quantised to INT8, powered by the CTranslate2 C++ inference library.

---

## What Has Been Built

The core of the project is a C++ translation engine wrapped in a class called `NMTWrapper`. When it starts up, it loads both the CTranslate2 model and a SentencePiece BPE tokeniser from disk. When asked to translate, it tokenises the input, appends the required NLLB language tags (`</s>` and `eng_Latn`), runs a greedy beam search with the Italian prefix (`ita_Latn`), decodes the output tokens back to text, and strips the leftover language tag before returning the final string.

[SCREENSHOT NEEDED: File: `src/NMT/NMTWrapper.cpp`, Lines: 35–94, Context: The full translate() pipeline — tokenise, tag, batch-translate, detokenise, cleanup.]

The engine is exposed in two ways. For development and evaluation on a Windows PC, there is a TCP server (`main.cpp`) built on Winsock2 that listens on port 18080, receives raw English text over the wire, and sends back the translated Italian string. For the actual HoloLens 2 deployment, the engine is compiled as a shared DLL (`NMT_MenKan_Plugin.dll`) through `UnityInterface.cpp`, which exports three simple C functions — `InitModel`, `TranslateText`, and `CleanupModel` — that a Unity `MonoBehaviour` script calls via P/Invoke.

[SCREENSHOT NEEDED: File: `src/UnityInterface.cpp`, Lines: 1–45, Context: The DLL export surface used by Unity on HoloLens 2.]

The build system (CMake) is set up to handle both targets automatically. On x64 it links against OpenBLAS; on ARM64 it switches to Google's Ruy backend, which provides native INT8 support for the Snapdragon 850 processor in the HoloLens 2.

---

## Data Processing and Evaluation Setup

- **Inference Pipeline:** The NMT engine processes raw English strings (primarily spoken dialogue captured from ASR) through a high-performance INT8 pipeline, which involves initial SentencePiece BPE tokenization, appending the required NLLB source language tags (`</s>` and `eng_Latn`), performing greedy beam search inference (`beam_size=1`) steered by an Italian target prefix (`ita_Latn`), and finally detokenizing the resulting byte-pair tokens back into human-readable text.

---

## Evaluation

The fast evaluation pipeline (`evaluate_nmt_fast.py`) was run on 25 March 2026, translating 1,000 sentence pairs from the **FLoRes-200 devtest** corpus — the official benchmark dataset used by Meta AI when publishing NLLB results. It ran in just over 100 seconds, processing sentences in batches of 32, and produced the following scores:

- **BLEU: 26.92**
- **chrF++: 56.16**

These are genuinely solid numbers. For context, a BLEU score in the **25–30 range** is generally considered good for a machine translation system on a professionally-translated benchmark — it means the outputs are fluent and largely correct, even if phrasing choices occasionally differ from the reference. The official Meta NLLB paper reports scores in a similar range for the distilled 600M model on high-resource European language pairs, so this result is consistent with what the model is expected to achieve. The chrF++ score of **56.16** provides a complementary view based on character-level overlap, which is particularly informative for Italian with its rich morphology, and it reinforces that the translations are structurally sound.

Looking at the sample outputs, the translations read naturally. For example, the sentence *"Like some other experts, he is skeptical about whether diabetes can be cured"* was rendered as *"Come alcuni altri esperti, egli è scettico sul fatto che il diabete possa essere curato"* — grammatically correct and idiomatic. The main differences from the reference translations are stylistic rather than factual, which is expected behaviour for a general-purpose NMT model not yet fine-tuned on spoken dialogue.

[SCREENSHOT NEEDED: File: `evaluation_report.txt`, Lines: 1–35, Context: Full evaluation output showing BLEU/chrF++ scores and sample English → Italian sentence pairs from the FLoRes-200 devtest run.]

The earlier TCP-based evaluation (`evaluate_nmt.py`) still returns 0.00 BLEU due to a receive buffer issue on the Python client side — this is a known bug unrelated to model quality and will be fixed separately.

---

## Latency (PC Testing)

Latency logs from four test runs show engine compute times ranging from roughly **3 seconds on short sentences up to 15 seconds on longer ones** with the current single-threaded configuration. An earlier multi-threaded run spiked to nearly 50 seconds and produced `BLAS: Bad memory unallocation` errors in the console. The single-threaded fix (one call to `ctranslate2::set_num_threads(1)`) resolved the instability. These numbers are expected to drop significantly on ARM64 with the Ruy backend.

[SCREENSHOT NEEDED: File: `server_latency.log`, Lines: 1–28, Context: Server console output showing per-sentence compute times under the single-threaded configuration.]

---

## Challenges

- **Language pair pivot (Turkish → Italian):** The project was originally scoped for English-to-Turkish translation, but Turkish word order (SOV — Subject-Object-Verb) created significant structural mismatches with English (SVO), causing the model to produce poorly ordered output even with a strong base model. Italian, sharing far more grammatical structure with English, was selected as the target language to keep the project moving while preserving the core NMT research goals.
- **OpenBLAS threading instability on Windows:** Enabling multi-threaded inference caused severe memory errors (`BLAS: Bad memory unallocation`) and latency spikes as high as 49 seconds per sentence. The fix required pinning CTranslate2 to a single thread, trading parallelism for stability — a known limitation of the OpenBLAS vcpkg build on Windows that will be resolved on ARM64 target using Google's Ruy backend.
- **Model hallucination from missing NLLB language tags:** Without appending `</s>` and `eng_Latn` after tokenization, the NLLB model would loop and hallucinate output (e.g., *"Holo Holo Holo..."*). This required understanding NLLB's mandatory source-language tagging format, which is not enforced by CTranslate2 itself.
- **TCP evaluation pipeline producing empty predictions:** The initial BLEU evaluation returned 0.00 because the Python client's `recv(1024)` buffer was too small to capture full translation responses, making it appear the engine was not translating at all. A separate direct-Python evaluation script (`evaluate_nmt_fast.py`) was built to bypass the TCP layer and obtain valid results.
- **ASR input quality:** Translation quality degrades when the upstream speech recogniser omits punctuation or sends short utterance fragments rather than full sentences — a pipeline-level challenge that the NMT layer alone cannot solve without a punctuation restoration step.

---

## What Is Still Remaining

The most immediate items are fixing the evaluation script's receive buffer so a real BLEU score can be obtained, and running the fast evaluator on FLoRes-200. Beyond that, the engine has not yet been deployed to a physical HoloLens 2 device for ARM64 testing. Fine-tuning on the OpenSubtitles corpus (to better handle informal spoken ASR output) is planned but not started. There are also two pipeline-level concerns noted during development: ASR output typically lacks punctuation, which affects translation quality, and translating short utterance fragments independently loses sentence context.

---

*All technical details in this report are sourced directly from the repository source files, build logs, and evaluation output.*
